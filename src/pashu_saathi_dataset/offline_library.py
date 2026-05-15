from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


LIBRARY_VERSION = "offline-library-v1"
DOWNLOAD_DATE = "2026-05-15"

ACCEPTED = "accepted"
ACCEPTED_STRIPPED = "accepted_stripped"
QUARANTINED = "quarantined"
REJECTED = "rejected"
DOWNLOAD_FAILED = "download_failed"

USER_AGENT = "PashuPulseSourceAcquisition/1.0 (+https://github.com/Rishavutkarsh/PashuPulse)"

RISKY_SPAN_PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s*(mg|ml|g|iu|mcg)\s*/\s*kg\b", re.IGNORECASE),
    re.compile(r"\b(intramuscular|subcutaneous|intravenous|i/v|s/c|inject|injection)\b", re.IGNORECASE),
    re.compile(r"\b(dose|dosage|route|withdrawal period|antibiotic|oxytocin|xylazine|diclofenac|sedative)\b", re.IGNORECASE),
    re.compile(r"\b(trocar|rumenotomy|lance|incision|stomach tube|calf puller|wire saw|castration|dehorning)\b", re.IGNORECASE),
]

HARD_REJECT_PATTERNS = [
    re.compile(r"\b(miracle cure|guaranteed cure|100% cure|secret remedy|natural antibiotic)\b", re.IGNORECASE),
    re.compile(r"\b(kerosene|engine oil|tobacco water|caustic|acid)\b", re.IGNORECASE),
    re.compile(r"\b(anti[- ]?vaccine|vaccines are useless|avoid veterinarian)\b", re.IGNORECASE),
    re.compile(r"\b(buy now|promo code|limited offer|affiliate link)\b", re.IGNORECASE),
]

MOJIBAKE_PATTERNS = [
    re.compile(r"(Ã|Â|â€™|â€œ|â€|�)"),
    re.compile(r"(.)\1{18,}"),
]

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(home|about us|privacy policy|terms of use|cookie policy|share this|print page|skip to content)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(copyright|all rights reserved|last updated).*$", re.IGNORECASE),
]

TOPIC_KEYWORDS = {
    "feeding": ["feed", "fodder", "ration", "nutrition", "roughage", "concentrate", "mineral"],
    "water": ["water", "drinking", "hydration"],
    "shed hygiene": ["shed", "housing", "manure", "bedding", "hygiene", "clean"],
    "milk hygiene": ["milk", "milking", "mastitis", "udder", "teat"],
    "heat/cold stress": ["heat", "cold", "stress", "shade", "ventilation"],
    "calf care": ["calf", "colostrum", "weaning"],
    "pregnancy/calving": ["pregnancy", "calving", "reproduction", "parturition"],
    "bloat": ["bloat", "rumen", "tympany"],
    "wounds": ["wound", "injury", "maggot", "fly"],
    "diarrhea": ["diarrhea", "diarrhoea", "scour", "dehydration"],
    "FMD-like signs": ["foot and mouth", "fmd", "vesicular", "mouth", "hoof"],
    "poisoning/spoiled feed": ["poison", "toxin", "aflatoxin", "mold", "mould"],
    "parasites": ["parasite", "tick", "worm", "fluke"],
    "bites": ["bite", "rabies", "snake", "dog"],
    "working ox care": ["draught", "draft", "bullock", "ox", "yoke", "work"],
}

SPECIES_KEYWORDS = {
    "cow": ["cow", "cows", "cattle", "dairy cattle"],
    "buffalo": ["buffalo", "buffaloes", "water buffalo"],
    "ox/bullock": ["ox", "oxen", "bullock", "draught", "draft"],
    "calf": ["calf", "calves", "heifer"],
}


@dataclass(frozen=True)
class OfflineSourceSpec:
    source_id: str
    title: str
    publisher: str
    url: str
    country: str
    source_tier: str
    language: str
    species: list[str]
    topics: list[str]
    geo_relevance: list[str]
    license_or_terms: str
    notes: str = ""


def source_catalog() -> list[OfflineSourceSpec]:
    specs = [
        # India anchors.
        ("dahd_lhdcp", "Livestock Health and Disease Control Programme", "Department of Animal Husbandry and Dairying, Government of India", "https://www.dahd.gov.in/schemes-programmes/lh-dc", "India", "official_india", "en", ["cow", "buffalo"], ["animal health", "extension", "escalation"], ["India"]),
        ("dahd_nadcp", "National Animal Disease Control Programme", "Department of Animal Husbandry and Dairying, Government of India", "https://www.dahd.gov.in/schemes/programmes/nadcp", "India", "official_india", "en", ["cow", "buffalo"], ["FMD-like signs", "brucellosis", "movement control"], ["India"]),
        ("nddb_good_dairy_husbandry", "Handbook of Good Dairy Husbandry Practices", "National Dairy Development Board", "https://www.nddb.coop/sites/default/files/pdfs/Handbook-of-Good-Dairy-Husbandry-Practices.pdf", "India", "official_extension_india", "en", ["cow", "buffalo", "calf"], ["feeding", "shed hygiene", "milk hygiene", "health"], ["India"]),
        ("nddb_water", "Importance of Drinking Water for Dairy Animals", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-nutrition/importance-of-drinking-water-for-dairy-animals", "India", "official_extension_india", "en", ["cow", "buffalo"], ["water", "heat/cold stress"], ["India"]),
        ("nddb_calf_nutrition", "Calf Nutrition", "National Dairy Development Board", "https://www.nddb.coop/node/511", "India", "official_extension_india", "en", ["calf", "cow", "buffalo"], ["calf care", "feeding"], ["India"]),
        ("nddb_clean_milk", "Clean Milk Production Awareness", "National Dairy Development Board", "https://www.nddb.coop/services/qa/capacity-building", "India", "official_extension_india", "en", ["cow", "buffalo"], ["milk hygiene"], ["India"]),
        ("nddb_animal_nutrition", "Animal Nutrition", "National Dairy Development Board", "https://www.nddb.coop/services/animalnutrition", "India", "official_extension_india", "en", ["cow", "buffalo"], ["feeding", "nutrition"], ["India"]),
        ("nddb_ration_balancing", "Ration Balancing Programme", "National Dairy Development Board", "https://www.nddb.coop/services/animalnutrition/programmes/ration-balancing-programme", "India", "official_extension_india", "en", ["cow", "buffalo"], ["feeding", "nutrition"], ["India"]),
        ("nddb_cattle_feed", "Compound Cattle Feed", "National Dairy Development Board", "https://www.nddb.coop/services/animalnutrition/cattlefeed", "India", "official_extension_india", "en", ["cow", "buffalo", "calf"], ["feeding"], ["India"]),
        ("nddb_mastitis", "Mastitis", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-health/disease/mastitis", "India", "official_extension_india", "en", ["cow", "buffalo"], ["milk hygiene", "mastitis"], ["India"]),
        ("nddb_bloat", "Bloat", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-health/disease/others/bloat", "India", "official_extension_india", "en", ["cow", "buffalo"], ["bloat", "feeding"], ["India"]),
        ("nddb_aflatoxicosis", "Aflatoxicosis", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-health/disease/fungal/Aflatoxicosis", "India", "official_extension_india", "en", ["cow", "buffalo"], ["poisoning/spoiled feed", "feeding"], ["India"]),
        ("nddb_vaccination_schedule", "Vaccination Schedules for Cattle and Buffalo", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-health/vaccination/schedules", "India", "official_extension_india", "en", ["cow", "buffalo"], ["animal health", "bites"], ["India"]),
        ("nddb_fly_control", "Fly Control", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-health/general/fly-control", "India", "official_extension_india", "en", ["cow", "buffalo"], ["shed hygiene", "wounds"], ["India"]),
        ("nddb_zoonosis_gi", "Gastro-intestinal Zoonotic Diseases", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-health/zoonosis/gastro-intestinal-zoonotic-diseases", "India", "official_extension_india", "en", ["cow", "buffalo", "calf"], ["diarrhea", "hygiene"], ["India"]),
        ("nddb_brucellosis", "Brucellosis", "National Dairy Development Board", "https://www.nddb.coop/farmer/animal-health/disease/bacterial/Brucellosis", "India", "official_extension_india", "en", ["cow", "buffalo"], ["pregnancy/calving", "animal health"], ["India"]),
        ("nddb_surra_hi", "Trypanosomiasis Surra", "National Dairy Development Board", "https://www.nddb.coop/hi/node/1081", "India", "official_extension_india", "hi", ["cow", "buffalo"], ["parasites", "animal health"], ["India"]),
        ("tnau_calf_management", "Calf Management", "Tamil Nadu Agricultural University Agritech Portal", "https://www.agritech.tnau.ac.in/expert_system/cattlebuffalo/Calf%20management.html", "India", "official_extension_india", "en", ["cow", "buffalo", "calf"], ["calf care", "pregnancy/calving"], ["India"]),
        ("tnau_general_care", "General Care and Management", "Tamil Nadu Agricultural University Agritech Portal", "https://www.agritech.tnau.ac.in/expert_system/cattlebuffalo/General%20care%20and%20management.html", "India", "official_extension_india", "en", ["cow", "buffalo", "calf"], ["calf care", "feeding", "shed hygiene"], ["India"]),
        ("vikaspedia_livestock", "Livestock Portal", "Vikaspedia", "https://vikaspedia.in/agriculture/livestock", "India", "secondary_context", "en", ["cow", "buffalo", "calf", "ox/bullock"], ["farmer language", "husbandry"], ["India"]),
        ("vikaspedia_calves", "Rearing of Calves", "Vikaspedia", "https://en.vikaspedia.in/viewcontent/agriculture/livestock/cattle-buffalo/rearing-of-calves?lgn=en", "India", "secondary_context", "en", ["calf", "cow", "buffalo"], ["calf care"], ["India"]),
        ("vikaspedia_housing", "Housing Management of Cattle and Buffalo", "Vikaspedia", "https://en.vikaspedia.in/viewcontent/agriculture/livestock/cattle-buffalo/housing-management-of-cattle-buffalo?lgn=en", "India", "secondary_context", "en", ["cow", "buffalo"], ["shed hygiene", "housing"], ["India"]),
        # FAO and global manuals.
        ("fao_dairy_index", "Small-Scale Dairy Farming Manual Index", "Food and Agriculture Organization", "https://www.fao.org/4/t1265e/t1265e.htm", "International", "international_extension", "en", ["cow", "buffalo", "calf", "ox/bullock"], ["dairy husbandry"], ["International", "South Asia", "Tropics"]),
        ("fao_dairy_housing", "Dairy Cattle and Buffalo Housing", "Food and Agriculture Organization", "https://www.fao.org/4/t1265e/t1270e05.htm", "International", "international_extension", "en", ["cow", "buffalo", "calf"], ["shed hygiene", "housing"], ["International", "Tropics"]),
        ("fao_dairy_feeding", "Feeding Dairy Cattle and Buffalo", "Food and Agriculture Organization", "https://www.fao.org/4/t1265e/t1275e01.htm", "International", "international_extension", "en", ["cow", "buffalo", "calf"], ["feeding", "nutrition"], ["International", "Tropics"]),
        ("fao_dairy_reproduction", "Reproduction in Dairy Cattle and Buffalo", "Food and Agriculture Organization", "https://www.fao.org/3/t1265e/t1280e01.htm", "International", "international_extension", "en", ["cow", "buffalo"], ["pregnancy/calving", "reproduction"], ["International", "Tropics"]),
        ("fao_dairy_disease", "Important Conditions Affecting Dairy Cattle and Buffalo", "Food and Agriculture Organization", "https://www.fao.org/4/t1265e/t1285e01.htm", "International", "international_extension", "en", ["cow", "buffalo"], ["animal health", "bloat", "wounds", "FMD-like signs"], ["International", "Tropics"]),
        ("fao_dairy_parasites", "Parasites in Dairy Cattle and Buffalo", "Food and Agriculture Organization", "https://www.fao.org/4/t1265e/t1285e06.htm", "International", "international_extension", "en", ["cow", "buffalo", "calf"], ["parasites"], ["International", "Tropics"]),
        ("fao_dairy_first_aid", "Farmer First Aid", "Food and Agriculture Organization", "https://www.fao.org/4/t1265e/t1285e09.htm", "International", "international_extension", "en", ["cow", "buffalo"], ["animal health", "bloat", "wounds", "pregnancy/calving"], ["International", "Tropics"]),
        ("fao_good_dairy_practice_pdf", "Guide to Good Dairy Farming Practice", "FAO and International Dairy Federation", "https://www.fao.org/4/ba0027e/ba0027e.pdf", "International", "international_extension", "en", ["cow", "buffalo"], ["milk hygiene", "feeding", "welfare"], ["International"]),
        ("fao_buffaloes", "Buffaloes", "Food and Agriculture Organization", "https://www.fao.org/dairy-production-products/dairy/buffaloes/en", "International", "international_extension", "en", ["buffalo"], ["buffalo", "milk production"], ["International", "India"]),
        ("fao_smallholder_dairying", "Smallholder Dairying in the Tropics", "CGIAR/ILRI", "https://cgspace.cgiar.org/server/api/core/bitstreams/23dbf2f1-cef1-4f20-9f6c-5f072b6020d3/content", "International", "international_extension", "en", ["cow", "buffalo", "calf"], ["feeding", "smallholder"], ["South Asia", "Tropics"]),
        ("cgiar_climate_smart_farmers_manual", "Climate-Smart Agriculture Farmers Manual", "CGIAR", "https://cgspace.cgiar.org/items/cf2350b0-93ca-4efd-a024-29d9497ff3fd", "International", "international_extension", "en", ["cow", "calf"], ["heat/cold stress", "feeding", "farm operations"], ["Africa", "Tropics"]),
        ("woah_terrestrial_code", "WOAH Terrestrial Code", "World Organisation for Animal Health", "https://www.woah.org/en/what-we-do/standards/codes-and-manuals/terrestrial-code-online-access/", "International", "international_standard", "en", ["cow", "buffalo"], ["welfare", "disease control"], ["International"]),
        ("woah_dairy_welfare_2014", "Animal Welfare and Dairy Cattle Production Systems Draft", "World Organisation for Animal Health", "https://www.woah.org/app/uploads/2021/03/a-tahsc-feb-2014-part-b.pdf", "International", "international_standard", "en", ["cow", "calf"], ["welfare", "housing"], ["International"]),
        # University extension and reputable references.
        ("umn_heat_dairy", "Heat Stress in Dairy Cattle", "University of Minnesota Extension", "https://extension.umn.edu/dairy-milking-cows/heat-stress-dairy-cattle", "United States", "university_extension", "en", ["cow"], ["heat/cold stress", "water"], ["International"]),
        ("umn_heat_feedlot", "Managing Heat Stress in Feedlot Cattle", "University of Minnesota Extension", "https://extension.umn.edu/beef-feedlot/managing-heat-stress-feedlot-cattle", "United States", "university_extension", "en", ["cow"], ["heat/cold stress", "water"], ["International"]),
        ("umn_dairy", "Dairy Extension Resources", "University of Minnesota Extension", "https://extension.umn.edu/dairy", "United States", "university_extension", "en", ["cow", "calf"], ["feeding", "milk hygiene", "calf care"], ["International"]),
        ("penn_state_newborn_calf", "Feeding the Newborn Dairy Calf", "Penn State Extension", "https://extension.psu.edu/feeding-the-newborn-dairy-calf", "United States", "university_extension", "en", ["calf"], ["calf care", "feeding"], ["International"]),
        ("penn_state_dairy", "Dairy Extension Articles", "Penn State Extension", "https://extension.psu.edu/animals-and-livestock/dairy", "United States", "university_extension", "en", ["cow", "calf"], ["feeding", "housing", "milk hygiene"], ["International"]),
        ("dairexnet_home", "DAIReXNET", "Extension Foundation", "https://dairy-cattle.extension.org/", "United States", "university_extension", "en", ["cow", "calf"], ["dairy husbandry"], ["International"]),
        ("dairexnet_rumen_calf", "Rumen Development in the Dairy Calf", "Extension Foundation", "https://dairy-cattle.extension.org/rumen-development-in-the-dairy-calf/", "United States", "university_extension", "en", ["calf"], ["calf care", "feeding"], ["International"]),
        ("dairexnet_calf_heifer", "Dairy Calf and Dairy Heifer Management", "Extension Foundation", "https://dairy-cattle.extension.org/dairy-calf-and-dairy-heifer-management/", "United States", "university_extension", "en", ["calf", "cow"], ["calf care", "housing"], ["International"]),
        ("dairexnet_calf_nutrition", "Raising Dairy Replacements: Calf Nutrition", "Extension Foundation", "https://dairy-cattle.extension.org/raising-dairy-replacementscalf-nutrition/", "United States", "university_extension", "en", ["calf"], ["calf care", "feeding"], ["International"]),
        ("dairexnet_electrolytes", "Electrolytes for Dairy Calves", "Extension Foundation", "https://dairy-cattle.extension.org/electrolytes-for-dairy-calves/", "United States", "university_extension", "en", ["calf"], ["diarrhea", "calf care"], ["International"]),
        ("animal_welfare_calf_transport", "Transport and Care of Calves", "Extension Foundation", "https://animal-welfare.extension.org/transport-and-care-of-calves/", "United States", "university_extension", "en", ["calf"], ["calf care", "welfare"], ["International"]),
        ("beef_brd_factsheets", "Bovine Respiratory Disease Factsheets", "Extension Foundation", "https://beef-cattle.extension.org/bovine-respiratory-disease-factsheets/", "United States", "university_extension", "en", ["cow", "calf"], ["animal health", "calf care"], ["International"]),
        ("beef_drylot", "Drylot Beef Cow Calf Production", "Extension Foundation", "https://beef-cattle.extension.org/drylot-beef-cow-calf-production/", "United States", "university_extension", "en", ["cow", "calf"], ["feeding", "housing"], ["International"]),
        ("msd_bloat", "Bloat in Ruminants", "MSD Veterinary Manual", "https://www.msdvetmanual.com/digestive-system/diseases-of-the-ruminant-forestomach/bloat-in-ruminants", "International", "reputable_veterinary_reference", "en", ["cow", "buffalo"], ["bloat"], ["International"]),
        ("msd_mastitis", "Mastitis in Cattle", "MSD Veterinary Manual", "https://www.msdvetmanual.com/reproductive-system/mastitis-in-large-animals/mastitis-in-cattle", "International", "reputable_veterinary_reference", "en", ["cow"], ["milk hygiene", "mastitis"], ["International"]),
        ("msd_intestinal_cattle", "Intestinal Diseases in Cattle", "MSD Veterinary Manual", "https://www.msdvetmanual.com/digestive-system/intestinal-diseases-in-ruminants/intestinal-diseases-in-cattle", "International", "reputable_veterinary_reference", "en", ["cow", "calf"], ["diarrhea", "animal health"], ["International"]),
        ("msd_nutrition_disorders", "Nutrition-Related Disorders in Beef Cattle", "MSD Veterinary Manual", "https://www.msdvetmanual.com/management-and-nutrition/nutrition-beef-cattle/prevention-of-common-nutrition-related-disorders-in-beef-cattle", "International", "reputable_veterinary_reference", "en", ["cow"], ["feeding", "bloat"], ["International"]),
        ("msd_udder_disorders", "Physiologic Disorders of the Udder in Cows", "MSD Veterinary Manual", "https://www.msdvetmanual.com/reproductive-system/udder-diseases-in-cows/physiologic-disorders-of-the-udder-in-cows", "International", "reputable_veterinary_reference", "en", ["cow"], ["milk hygiene"], ["International"]),
        ("msd_buffalo_parasite_image", "Water Buffalo Calf Parasite Image Context", "MSD Veterinary Manual", "https://www.msdvetmanual.com/multimedia/image/necropsy-photo-toxocara-vitulorum-infestation-water-buffalo", "International", "reputable_veterinary_reference", "en", ["buffalo", "calf"], ["parasites", "diarrhea"], ["International"]),
        # India research/context sources.
        ("icar_buffalo_livelihood", "Buffaloes: Promising Livestock for Rural Livelihood Improvement", "Indian Farming / ICAR ePubs", "https://epubs.icar.org.in/index.php/IndFarm/article/view/84649", "India", "academic_extension", "en", ["buffalo", "ox/bullock"], ["buffalo", "feeding", "farm operations", "working ox care"], ["India"]),
        ("icar_buffalo_adoption", "Buffalo Husbandry Adoption Study", "Indian Journal of Extension Education / ICAR ePubs", "https://epubs.icar.org.in/index.php/IJEE/article/view/128536", "India", "academic_extension", "en", ["buffalo"], ["farm operations", "housing", "breeding"], ["India"]),
        ("icar_extension_constraints", "Animal Husbandry Extension Constraints", "Journal of Agricultural Extension Management / ICAR ePubs", "https://epubs.icar.org.in/index.php/JAEM/article/view/105107", "India", "academic_extension", "en", ["cow", "buffalo"], ["extension", "farm operations"], ["India"]),
        # Hindi and additional Indian dairy context.
        ("nddb_hi_animal_nutrition", "पशु पोषण", "National Dairy Development Board", "https://www.nddb.coop/hi/services/animalnutrition", "India", "official_extension_india", "hi", ["cow", "buffalo"], ["feeding", "nutrition"], ["India"]),
        ("nddb_hi_ration_balance", "आहार संतुलन कार्यक्रम", "National Dairy Development Board", "https://www.nddb.coop/hi/services/animalnutrition/rationbalance", "India", "official_extension_india", "hi", ["cow", "buffalo", "calf"], ["feeding", "nutrition"], ["India"]),
        ("nddb_hi_new_feed", "नया पशु आहार", "National Dairy Development Board", "https://www.nddb.coop/hi/services/animalnutrition/cattlefeed/new-feed-variants", "India", "official_extension_india", "hi", ["buffalo", "cow"], ["feeding", "buffalo"], ["India"]),
        ("nddb_hi_reproduction", "पशु प्रजनन", "National Dairy Development Board", "https://www.nddb.coop/hi/services/animalbreeding/animalreproduction", "India", "official_extension_india", "hi", ["cow", "buffalo"], ["pregnancy/calving", "reproduction"], ["India"]),
        ("nddb_hi_vaccination_schedule", "टीकाकरण अनुसूची", "National Dairy Development Board", "https://www.nddb.coop/hi/node/1127", "India", "official_extension_india", "hi", ["cow", "buffalo", "calf"], ["FMD-like signs", "bites", "animal health"], ["India"]),
        ("nddb_climate_smart_dairying", "Climate Smart Dairying", "National Dairy Development Board", "https://www.nddb.coop/services/animalnutrition/climate-smart-dairying", "India", "official_extension_india", "en", ["cow", "buffalo"], ["feeding", "water", "farm operations"], ["India"]),
        # Additional university extension coverage.
        ("wisc_calf_cold_stress", "Cold Stress in Dairy Calves", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/cold-stress-in-dairy-calves/", "United States", "university_extension", "en", ["calf"], ["calf care", "heat/cold stress", "shed hygiene"], ["International"]),
        ("wisc_calf_heat_stress", "Managing Heat in Pre-Weaned Calves", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/managing-the-heat-in-pre-weaned-calves/", "United States", "university_extension", "en", ["calf"], ["calf care", "heat/cold stress", "water"], ["International"]),
        ("wisc_dairy_heat_mammary", "Heat Stress in Mammary Gland Development and Health", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/impact-of-heat-stress-in-mammary-gland-development-and-health-in-dairy-cows/", "United States", "university_extension", "en", ["cow", "calf"], ["heat/cold stress", "milk hygiene"], ["International"]),
        ("wisc_dairy_summer_ration", "Dairy Cattle Ration Management During Summer", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/dairy-cattle-ration-management-during-summer/", "United States", "university_extension", "en", ["cow"], ["feeding", "heat/cold stress"], ["International"]),
        ("wisc_dry_cow_heat", "Dry Cow Heat Stress Management", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/dry-cow-heat-stress-management/", "United States", "university_extension", "en", ["cow", "calf"], ["heat/cold stress", "pregnancy/calving"], ["International"]),
        ("wisc_grazing_heat", "Dealing with Hot Weather in Grazing Systems", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/dealing-with-hot-weather-in-grazing-systems/", "United States", "university_extension", "en", ["cow"], ["heat/cold stress", "water", "feeding"], ["International"]),
        ("wisc_animal_handling_heat", "Animal Handling During Heat Stress", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/animal-handling-during-heat-stress/", "United States", "university_extension", "en", ["cow"], ["heat/cold stress", "water", "welfare"], ["International"]),
        ("wisc_heat_facilities", "Heat Stress Abatement in Dairy Facilities", "University of Wisconsin Dairy Extension", "https://dairy.extension.wisc.edu/articles/heat-stress-abatement-in-dairy-facilities/", "United States", "university_extension", "en", ["cow", "calf"], ["heat/cold stress", "housing", "welfare"], ["International"]),
        # Ox/bullock and draught animal sources.
        ("fao_draught_animal_power", "Draught Animal Power and Implements", "Food and Agriculture Organization", "https://www.fao.org/family-farming/detail/en/c/1619223/", "International", "international_extension", "en", ["ox/bullock", "buffalo", "cow"], ["working ox care", "welfare", "farm operations"], ["International", "Tropics"]),
        ("fao_draught_performance", "Draught Performance", "Food and Agriculture Organization", "https://www.fao.org/4/AD347E/ad347e0g.htm", "International", "international_extension", "en", ["ox/bullock", "cow"], ["working ox care", "welfare"], ["International"]),
        ("aciar_draught_animals", "Draught Animals in Farming Systems", "Australian Centre for International Agricultural Research", "https://www.aciar.gov.au/sites/default/files/legacy/node/2123/pr27_pdf_72210.pdf", "International", "international_extension", "en", ["ox/bullock", "cow", "buffalo"], ["working ox care", "farm operations", "welfare"], ["International", "Tropics"]),
    ]
    return [
        OfflineSourceSpec(
            source_id=source_id,
            title=title,
            publisher=publisher,
            url=url,
            country=country,
            source_tier=tier,
            language=language,
            species=species,
            topics=topics,
            geo_relevance=geo,
            license_or_terms="Public web source; verify reuse terms before redistribution.",
        )
        for source_id, title, publisher, url, country, tier, language, species, topics, geo in specs
    ]


def build_offline_library(out_dir: Path, max_sources: int | None = None, skip_download: bool = False) -> dict[str, Any]:
    raw_dir = out_dir / "raw"
    extracted_dir = out_dir / "extracted_text"
    clean_dir = out_dir / "clean_text"
    manifests_dir = out_dir / "manifests"
    reports_dir = out_dir / "reports"
    for directory in [raw_dir, extracted_dir, clean_dir, manifests_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    specs = source_catalog()
    if max_sources:
        specs = specs[:max_sources]

    download_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    clean_chunk_rows: list[dict[str, Any]] = []

    seen_urls: set[str] = set()
    for spec in specs:
        if spec.url in seen_urls:
            quality_rows.append(_quality_row(spec, REJECTED, ["duplicate_url"], None, None, None, 0, 0))
            continue
        seen_urls.add(spec.url)
        raw_path = raw_dir / _raw_filename(spec)
        download = _download_source(spec, raw_path, skip_download)
        download_rows.append(download)
        if download["status"] != "downloaded":
            quality_rows.append(_quality_row(spec, DOWNLOAD_FAILED, ["download_failed"], raw_path, None, None, 0, 0))
            continue

        raw_bytes = raw_path.read_bytes()
        extracted_text = _extract_text(raw_path, raw_bytes, download["content_type"])
        extracted_path = extracted_dir / f"{spec.source_id}.txt"
        extracted_path.write_text(extracted_text, encoding="utf-8")
        clean_text, removed_spans = clean_source_text(extracted_text)
        clean_path = clean_dir / f"{spec.source_id}.txt"
        clean_path.write_text(clean_text, encoding="utf-8")

        raw_tokens = _token_estimate(extracted_text)
        clean_tokens = _token_estimate(clean_text)
        risk_flags = _risk_flags(spec, extracted_text, clean_text, removed_spans)
        status = _status_for_source(spec, extracted_text, clean_text, risk_flags, removed_spans)

        extraction_rows.append(
            {
                **_spec_metadata(spec),
                "downloaded_path": _rel(out_dir, raw_path),
                "extracted_path": _rel(out_dir, extracted_path),
                "clean_path": _rel(out_dir, clean_path),
                "content_type": download["content_type"],
                "raw_sha256": _sha256_bytes(raw_bytes),
                "extracted_sha256": _sha256_text(extracted_text),
                "clean_sha256": _sha256_text(clean_text),
                "raw_tokens": raw_tokens,
                "clean_tokens": clean_tokens,
                "removed_span_count": len(removed_spans),
            }
        )
        quality_rows.append(_quality_row(spec, status, risk_flags, raw_path, extracted_path, clean_path, raw_tokens, clean_tokens))
        if status in {ACCEPTED, ACCEPTED_STRIPPED}:
            clean_chunk_rows.extend(_chunk_source(spec, clean_text, clean_path, status))

    _write_jsonl(manifests_dir / "source_download_manifest.jsonl", download_rows)
    _write_jsonl(manifests_dir / "source_extraction_manifest.jsonl", extraction_rows)
    _write_jsonl(manifests_dir / "source_quality_manifest.jsonl", quality_rows)
    _write_jsonl(out_dir / "accepted_cpt_sources.jsonl", [row for row in quality_rows if row["status"] == ACCEPTED])
    _write_jsonl(out_dir / "accepted_stripped_cpt_sources.jsonl", [row for row in quality_rows if row["status"] == ACCEPTED_STRIPPED])
    _write_jsonl(out_dir / "quarantined_cpt_sources.jsonl", [row for row in quality_rows if row["status"] == QUARANTINED])
    _write_jsonl(out_dir / "rejected_cpt_sources.jsonl", [row for row in quality_rows if row["status"] in {REJECTED, DOWNLOAD_FAILED}])
    _write_jsonl(out_dir / "cpt_clean_chunks.jsonl", clean_chunk_rows)

    coverage = _coverage_report(quality_rows, clean_chunk_rows)
    safety = _safety_report(quality_rows)
    _write_json(reports_dir / "source_coverage_report.json", coverage)
    _write_json(reports_dir / "source_safety_filter_report.json", safety)
    manifest = _library_manifest(out_dir, quality_rows, clean_chunk_rows)
    _write_json(out_dir / "offline_library_manifest.json", manifest)
    return manifest


def clean_source_text(text: str) -> tuple[str, list[str]]:
    text = html.unescape(text).replace("\xa0", " ")
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            continue
        if any(pattern.search(stripped) for pattern in BOILERPLATE_PATTERNS):
            continue
        lines.append(stripped)
    sentences = _sentences(" ".join(lines))
    kept = []
    removed = []
    for sentence in sentences:
        if any(pattern.search(sentence) for pattern in HARD_REJECT_PATTERNS):
            removed.append(sentence)
            continue
        if any(pattern.search(sentence) for pattern in RISKY_SPAN_PATTERNS):
            removed.append(sentence)
            continue
        kept.append(sentence)
    clean = re.sub(r"\s+", " ", " ".join(kept)).strip()
    return clean, removed


def _download_source(spec: OfflineSourceSpec, raw_path: Path, skip_download: bool) -> dict[str, Any]:
    base = {
        **_spec_metadata(spec),
        "downloaded_path": _rel(raw_path.parents[2], raw_path) if len(raw_path.parents) > 2 else str(raw_path),
        "download_date": DOWNLOAD_DATE,
    }
    if skip_download:
        return {**base, "status": "skipped", "content_type": "unknown", "error": "skip_download"}
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read()
            content_type = response.headers.get("content-type", "application/octet-stream").split(";")[0].strip().lower()
        raw_path.write_bytes(data)
        time.sleep(0.05)
        return {
            **base,
            "status": "downloaded",
            "content_type": content_type or _content_type_from_path(raw_path),
            "raw_sha256": _sha256_bytes(data),
            "bytes": len(data),
            "error": "",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {**base, "status": "failed", "content_type": "unknown", "raw_sha256": "", "bytes": 0, "error": str(exc)}


def _extract_text(raw_path: Path, raw_bytes: bytes, content_type: str) -> str:
    suffix = raw_path.suffix.lower()
    if "pdf" in content_type or suffix == ".pdf":
        text = _extract_pdf_text(raw_path)
        if text.strip():
            return text
        return _decode_bytes(raw_bytes)
    decoded = _decode_bytes(raw_bytes)
    if "html" in content_type or suffix in {".html", ".htm"} or "<html" in decoded[:500].lower():
        return _html_to_text(decoded)
    return decoded


def _extract_pdf_text(raw_path: Path) -> str:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(raw_path))
            pages = []
            for index, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"\n[page {index + 1}]\n{page_text}")
            return "\n".join(pages)
        except Exception:
            continue
    return ""


def _html_to_text(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", markup)
    markup = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", markup)
    markup = re.sub(r"(?i)</\s*(p|div|li|h[1-6]|tr|section|article)\s*>", "\n", markup)
    markup = re.sub(r"(?s)<[^>]+>", " ", markup)
    text = html.unescape(markup)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _status_for_source(spec: OfflineSourceSpec, extracted: str, clean: str, risk_flags: list[str], removed_spans: list[str]) -> str:
    if "low_quality_text" in risk_flags:
        return REJECTED
    if "hard_reject_pattern" in risk_flags and spec.source_tier in {"commercial", "unsafe_blog", "low_trust_blog"}:
        return REJECTED
    if _token_estimate(clean) < 80:
        return QUARANTINED
    if len(removed_spans) > 12 or ("procedure_or_dose_heavy" in risk_flags and spec.source_tier not in {"official_india", "official_extension_india", "international_extension", "university_extension"}):
        return QUARANTINED
    if removed_spans or "hard_reject_pattern" in risk_flags or "non_target_species" in risk_flags:
        return ACCEPTED_STRIPPED
    return ACCEPTED


def _risk_flags(spec: OfflineSourceSpec, extracted: str, clean: str, removed_spans: list[str]) -> list[str]:
    flags = []
    if any(pattern.search(extracted) for pattern in HARD_REJECT_PATTERNS):
        flags.append("hard_reject_pattern")
    if any(pattern.search(extracted) for pattern in MOJIBAKE_PATTERNS):
        flags.append("possible_mojibake_or_ocr_noise")
    if len(removed_spans) > 0:
        flags.append("risky_spans_stripped")
    if len(removed_spans) > 12:
        flags.append("procedure_or_dose_heavy")
    lower = extracted.lower()
    if not any(term in lower for terms in SPECIES_KEYWORDS.values() for term in terms):
        flags.append("non_target_species")
    if _token_estimate(clean) < 80:
        flags.append("too_little_clean_text")
    if spec.language == "hi" and not re.search(r"[\u0900-\u097F]", extracted):
        flags.append("hindi_source_without_devanagari_detected")
    return flags


def _chunk_source(spec: OfflineSourceSpec, clean_text: str, clean_path: Path, status: str) -> list[dict[str, Any]]:
    words = clean_text.split()
    chunks = []
    size = 750
    overlap = 75
    start = 0
    index = 1
    while start < len(words):
        end = min(len(words), start + size)
        chunk_text = " ".join(words[start:end])
        if len(chunk_text.split()) >= 80:
            chunks.append(
                {
                    "chunk_id": f"{spec.source_id}_chunk_{index:03d}",
                    "source_id": spec.source_id,
                    "text": chunk_text,
                    "token_estimate": _token_estimate(chunk_text),
                    "status": status,
                    "species": spec.species,
                    "topics": spec.topics,
                    "language": spec.language,
                    "source_tier": spec.source_tier,
                    "clean_path": str(clean_path),
                    "content_hash": _sha256_text(chunk_text),
                    "allowed_for": ["cpt_dapt", "offline_retrieval_candidate"],
                    "not_allowed_for": ["sft_grounding_without_later_review"],
                }
            )
            index += 1
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _quality_row(
    spec: OfflineSourceSpec,
    status: str,
    risk_flags: list[str],
    raw_path: Path | None,
    extracted_path: Path | None,
    clean_path: Path | None,
    raw_tokens: int,
    clean_tokens: int,
) -> dict[str, Any]:
    return {
        **_spec_metadata(spec),
        "status": status,
        "risk_flags": risk_flags,
        "downloaded_path": str(raw_path) if raw_path else "",
        "extracted_path": str(extracted_path) if extracted_path else "",
        "clean_path": str(clean_path) if clean_path else "",
        "raw_tokens": raw_tokens,
        "clean_tokens": clean_tokens,
        "notes": spec.notes,
    }


def _coverage_report(quality_rows: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_rows = [row for row in quality_rows if row["status"] in {ACCEPTED, ACCEPTED_STRIPPED}]
    topic_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    language_counts = Counter(row["language"] for row in accepted_rows)
    tier_counts = Counter(row["source_tier"] for row in accepted_rows)
    country_counts = Counter(row["country"] for row in accepted_rows)
    for row in accepted_rows:
        topic_counts.update(row["topics"])
        species_counts.update(row["species"])
    required_topics = set(TOPIC_KEYWORDS)
    required_species = {"cow", "buffalo", "ox/bullock", "calf"}
    missing_topics = sorted(required_topics - set(topic_counts))
    missing_species = sorted(required_species - set(species_counts))
    clean_tokens = sum(row["clean_tokens"] for row in accepted_rows)
    source_family_tokens = Counter()
    for row in accepted_rows:
        source_family_tokens[row["publisher"]] += row["clean_tokens"]
    max_family_share = 0.0
    if clean_tokens:
        max_family_share = max(source_family_tokens.values(), default=0) / clean_tokens
    return {
        "valid": not missing_topics and not missing_species and len(accepted_rows) >= 25,
        "accepted_source_count": len(accepted_rows),
        "total_source_count": len(quality_rows),
        "chunk_count": len(chunks),
        "clean_tokens": clean_tokens,
        "topic_counts": dict(topic_counts),
        "species_counts": dict(species_counts),
        "language_counts": dict(language_counts),
        "tier_counts": dict(tier_counts),
        "country_counts": dict(country_counts),
        "missing_topics": missing_topics,
        "missing_species": missing_species,
        "max_source_family_token_share": round(max_family_share, 4),
        "coverage_goal": "coverage-first offline library; token count is secondary",
    }


def _safety_report(quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in quality_rows)
    risk_counts: Counter[str] = Counter()
    for row in quality_rows:
        risk_counts.update(row["risk_flags"])
    return {
        "valid": True,
        "status_counts": dict(status_counts),
        "risk_flag_counts": dict(risk_counts),
        "policy": {
            "accepted_keeps": ["husbandry", "nutrition", "symptom descriptions", "prevention", "welfare", "milk hygiene"],
            "stripped_or_quarantined": ["raw doses", "injection routes", "procedure walkthroughs", "withdrawal tables", "unsafe remedies"],
            "not_directly_allowed_for": ["SFT factual grounding", "automatic farmer-facing advice"],
        },
    }


def _library_manifest(out_dir: Path, quality_rows: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts = [
        "manifests/source_download_manifest.jsonl",
        "manifests/source_extraction_manifest.jsonl",
        "manifests/source_quality_manifest.jsonl",
        "reports/source_coverage_report.json",
        "reports/source_safety_filter_report.json",
        "accepted_cpt_sources.jsonl",
        "accepted_stripped_cpt_sources.jsonl",
        "quarantined_cpt_sources.jsonl",
        "rejected_cpt_sources.jsonl",
        "cpt_clean_chunks.jsonl",
    ]
    artifact_hashes = {}
    for artifact in artifacts:
        path = out_dir / artifact
        if path.exists():
            artifact_hashes[artifact] = _sha256_bytes(path.read_bytes())
    return {
        "library_version": LIBRARY_VERSION,
        "created_date": date.today().isoformat(),
        "status": "OFFLINE_SOURCE_LIBRARY_CANDIDATE",
        "source_count": len(quality_rows),
        "accepted_source_count": sum(1 for row in quality_rows if row["status"] in {ACCEPTED, ACCEPTED_STRIPPED}),
        "chunk_count": len(chunks),
        "clean_tokens": sum(row["clean_tokens"] for row in quality_rows if row["status"] in {ACCEPTED, ACCEPTED_STRIPPED}),
        "sft_allowed": False,
        "rag_grounding_allowed": False,
        "artifact_hashes": artifact_hashes,
        "notes": [
            "Offline preservation library for later CPT/retrieval selection.",
            "Not an approved SFT/RAG grounding set.",
            "Final grounding subset requires a separate review pass.",
        ],
    }


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?।])\s+", text) if part.strip()]


def _spec_metadata(spec: OfflineSourceSpec) -> dict[str, Any]:
    return {
        "source_id": spec.source_id,
        "title": spec.title,
        "publisher": spec.publisher,
        "url": spec.url,
        "download_date": DOWNLOAD_DATE,
        "content_type": "",
        "language": spec.language,
        "country": spec.country,
        "source_tier": spec.source_tier,
        "species": spec.species,
        "topics": spec.topics,
        "geo_relevance": spec.geo_relevance,
        "license_or_terms": spec.license_or_terms,
    }


def _raw_filename(spec: OfflineSourceSpec) -> str:
    extension = ".html"
    path = spec.url.split("?", 1)[0].lower()
    if path.endswith(".pdf"):
        extension = ".pdf"
    elif path.endswith(".txt"):
        extension = ".txt"
    return f"{spec.source_id}{extension}"


def _content_type_from_path(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "application/pdf"
    if path.suffix.lower() in {".html", ".htm"}:
        return "text/html"
    return "text/plain"


def _token_estimate(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
