from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METADATA_RESIDUE = re.compile(r"\b(mudda|field note|close note|review card|metadata|answer_shape|template|variant_axis)\b", re.I)
UNSAFE_TREATMENT = re.compile(
    r"\b(antibiotic|painkiller|dewormer|meloxicam|calcium|tablet|mg|ml|dose|injection|inject|puncture|cut|kerosene|alcohol|mustard oil|sarson ka tel|haldi|mirch|ash|force[- ]?feed|pull hard)\b",
    re.I,
)
NEGATION = re.compile(r"\b(mat|nahi|na karo|do not|don't|avoid|without|apni taraf se|on your own|kabhi nahi)\b", re.I)
ENGLISH_RESIDUE = re.compile(
    r"\b(ki|ka|ke|mein|nahi|mat|hatao|rakho|dekho|dikkat|bahut|zyada|kai|saath|beemar|phisalna|sujan|badbu|khada|chara|paani|saans|pet|ghav|khur|muh|doodh|bachde|gai|bhains|bail|lag raha)\b",
    re.I,
)
HINGLISH_WORDS = {
    "gai", "gaay", "bhains", "bail", "bachda", "bachde", "pashu", "paani", "chara", "doodh",
    "khana", "peena", "saans", "pet", "ghav", "khur", "muh", "sookha", "saaf", "alag",
    "shaant", "chhaya", "hawa", "aaraam", "dhyaan", "dekho", "rakho", "agar", "turant",
    "doctor", "sampark", "mat", "nahi", "bulao", "madad", "dast", "kamzor", "bukhar",
    "vaccine", "dewormer", "photo", "diagnosis", "trained", "guidance", "official", "affected",
    "behavior", "change", "sign", "safe", "check", "checks", "note", "routine", "mineral",
    "salt", "lick", "weight", "pot", "belly", "outbreak", "fever", "bleeding",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def topic_key(row: dict[str, Any]) -> str:
    seed = str(row.get("parent_seed_id") or row.get("row_id") or "")
    match = re.search(r"ps_v4_\d+_\d+_(.+)$", seed)
    return match.group(1) if match else str(row.get("family_bucket", "routine_care"))


def animal(row: dict[str, Any], lang: str) -> str:
    tags = set(row.get("tags", []))
    if "buffalo" in tags:
        return "bhains" if lang == "hinglish" else "buffalo"
    if "ox" in tags:
        return "bail" if lang == "hinglish" else "ox"
    if "calf" in tags:
        return "bachda" if lang == "hinglish" else "calf"
    return "gai" if lang == "hinglish" else "cow"


def topic_profile(key: str) -> dict[str, str]:
    exact = {
        "monsoon_ventilation": ("barsaat mein shed ki hawa", "shed mein hawa chalne do; bheed, nami aur badbu kam rakho", "saans dikkat, zyada nami, badbu, ya kai pashu beemar"),
        "gradual_feed_change": ("chara badalne ka time", "naya chara dheere introduce karo; purana aur naya chara ekdum se mix badalna mat", "pet phoolna, dast, bilkul na khana, ya susti"),
        "mineral_salt_awareness": ("namak-mineral ki routine", "mineral ya salt lick ko routine awareness tak rakho; dose ya dawa jaisa use mat karo", "na khana, kamzori, girna, ya sudden behavior change"),
        "calf_clean_bedding": ("bachde ka bedding", "bachde ko garam, saaf aur sookhe bedding par rakho", "thand lagna, dast, na peena, ya khada na ho paana"),
        "calf_suckling_observation": ("bachde ka doodh peena", "bachda maa ka doodh pee raha hai ya nahi dhyaan se dekho; zabardasti pilana mat", "na peena, bahut kamzori, dast, ya khada na hona"),
        "ox_rest_after_work": ("kaam ke baad bail ka aaraam", "kaam ke baad bail ko paani, chhaya aur aaraam do", "langdapan, saans phoolna, ghav, ya khada na hona"),
        "ox_yoke_fit_check": ("bail ke jua ka fit", "jua zyada tight ya ragad wala na ho; ghisav wali jagah saaf-sookhi rakho", "khula ghav, sujan, bleeding, ya bail kaam na kar paaye"),
        "hoof_routine_check": ("khur ki routine checking", "khur ke aas-paas gili mitti kam karo; chalna aur sujan dekho", "gehra ghav, tez langdapan, khada na hona, ya badbu"),
        "milk_utensil_cleaning": ("doodh ke bartan ki safai", "doodh nikalne se pehle bartan aur haath saaf rakho", "doodh mein khoon, clot, badbu, ya pashu ko bukhar"),
        "teat_cleaning_routine": ("thun ki safai", "doodh se pehle thun ko saaf rakho aur doodh mein badlav note karo", "udder garam, clot, khoon, dard, ya fever"),
        "shade_for_buffalo": ("bhains ko chhaya", "bhains ko chhaya, hawa aur thandi jagah do", "tez saans, girna, zyada susti, ya paani na peena"),
        "winter_calf_dryness": ("sardi mein bachde ko sookha rakhna", "bachde ko hawa se bachakar garam-sookhi jagah rakho", "kaapna, na peena, dast, ya bahut susti"),
        "new_animal_isolation": ("naye pashu ko alag rakhna", "naye pashu ko kuch din alag observe karo; turant herd mein mix mat karo", "khansi, dast, fever, muh/khur chhale, ya na khana"),
        "dung_appetite_log": ("dung aur bhook ka note", "khana, paani, dung aur behavior ka short note rakho", "bilkul na khana, khoon, girna, ya saans dikkat"),
        "market_return_observation": ("bazaar se lautne ke baad observation", "bazaar se aaye pashu ko observe karo; beemar sign ho to mixing aur sale rok do", "muh/khur chhale, fever, khansi, dast, ya kai pashu affected"),
        "calf_navel_clean_surroundings": ("bachde ki naabhi ke aas-paas safai", "naabhi ke aas-paas bedding saaf-sookha rakho; chedna ya kuch lagana mat", "sujan, badbu, peep, fever, ya bachda na peena"),
        "water_access_for_lactating": ("doodh wali pashu ka paani", "doodh wali pashu ko saaf paani aasani se milna chahiye", "paani na peena, doodh achanak kam, fever, ya susti"),
        "fly_control_shed": ("shed mein makkhiyan kam karna", "gandagi aur geela bedding hatao; shed saaf-sookha rakho", "ghav, maggot, badbu, ya kai pashu pareshaan"),
        "workload_rotation_ox": ("bail ka kaam baantna", "bail ko lagatar zyada kaam mat karvao; beech mein aaraam do", "langdapan, ghav, saans phoolna, ya kaam na kar paana"),
        "calf_clean_feeding_vessel": ("bachde ke doodh bartan ki safai", "bachde ka feeding bartan saaf rakho", "dast, na peena, pet phoolna, ya susti"),
        "rainy_season_fodder_check": ("barsaat mein chara check", "fungus, badbu ya geela-sadha chara hatao", "dast, kampan, girna, ya kai pashu beemar"),
        "mild_lameness": ("halka langdapan", "pashu ko aaraam do; khur, sujan aur ghav safe distance se dekho", "khada na hona, gehra ghav, tez sujan, ya severe dard"),
        "loose_dung_mild": ("halka patla dung", "saaf paani rakho; kharab chara hatao; dung aur appetite note karo", "bahut dast, khoon, aankh dhansna, ya khada na hona"),
        "reduced_milk_no_fever": ("bukhar ke bina doodh kam", "paani, chara, aaraam, milking routine aur udder safai check karo", "fever, clot, khoon, udder dard, ya pashu off-feed"),
        "yoke_rub_early": ("jua se halka ghisav", "jua ka fit sudharo; jagah saaf-sookhi rakho; bail ko aaraam do", "khula ghav, sujan, bleeding, ya badbu"),
        "eye_watering_dust": ("dhool se aankh paani", "dhool kam karo; pashu ko saaf hawa wali jagah rakho", "aankh band, sujan, peep, ya dikhai kam lagna"),
        "skin_itch_flies": ("makkhi se skin itch", "shed saaf rakho; makkhiyan kam karo; ghav ya sujan dekho", "khula ghav, bleeding, maggot, ya bahut zyada khujli"),
        "mild_belly_discomfort": ("pet mein halka discomfort", "kharab chara hatao; pet phoolna, dung aur appetite dekho", "pet tezi se phoolna, saans dikkat, girna, ya severe dard"),
        "post_transport_tired": ("safar ke baad thakan", "pashu ko paani, chhaya aur aaraam do; bheed se door rakho", "girna, saans dikkat, fever, ya na khana"),
        "ox_shoulder_soreness": ("bail ke kandhe ka dard", "kaam rok kar aaraam do; jua fit aur ghisav check karo", "sujan, khula ghav, bleeding, ya bail khada na ho"),
        "calf_not_drinking_well": ("bachda doodh/paani kam pee raha", "bachde ko garam-sookhi jagah rakho; na peene ka time note karo", "bilkul na peena, bahut kamzori, dast, ya khada na hona"),
        "mild_swelling_leg": ("pair mein halka sujan", "pashu ko aaraam do; sujan, dard aur chalna dekho", "sujan badhna, khada na hona, gehra ghav, ya fever"),
        "feed_refusal_after_change": ("chara badalne ke baad na khana", "naya chara rok kar safe purana chara aur paani do", "pet phoolna, bilkul na khana, dast, ya girna"),
        "mild_nasal_discharge": ("naak se halka paani", "dhool aur bheed kam karo; saans, khana aur fever dekho", "saans dikkat, fever, na khana, ya kai pashu affected"),
        "small_horn_scrape": ("singh ke paas halka scrape", "upar ki gandagi saaf paani se hatao; jagah saaf-sookhi rakho", "bleeding, sujan, badbu, ya gehra ghav"),
        "lactating_off_feed": ("doodh wali pashu ka khana chhodna", "paani, chara, doodh, udder aur fever dhyaan se dekho", "fever, doodh mein clot/khoon, pet phoolna, ya girna"),
        "multiple_animals_sick": ("kai pashu ek saath beemar", "affected pashuon ko alag rakho; movement aur bazaar le jaana rok do", "fever, chhale, dast, sudden death, ya breathing trouble"),
        "severe_breathing_trouble": ("saans ki severe dikkat", "pashu ko shaant, khuli hawa wali jagah rakho; bheed door rakho", "saans phoolna, girna, blue tongue, ya khada na ho paana"),
        "severe_milk_blood": ("doodh mein khoon ya clot", "doodh ka badlav note karo; udder ko saaf rakho; sale/ghar use par trained guidance lo", "fever, udder garam, khoon/clot, ya pashu off-feed"),
        "mustard_oil_bloat": ("pet phoolne par sarson tel pressure", "sarson tel ya drenching mat karo; suspect chara rok kar pashu ko shaant rakho", "pet tezi se phoolna, saans dikkat, girna, ya bechaini"),
        "random_antibiotic": ("random antibiotic pressure", "antibiotic apni taraf se mat do; sirf safe observation aur support karo", "fever, severe weakness, breathing trouble, ya halat bigadna"),
        "painkiller_for_work": ("kaam ke liye painkiller pressure", "painkiller mat do; bail ka kaam rok kar aaraam do", "langdapan, sujan, ghav, ya khada na ho paana"),
        "wound_chilli_ash": ("ghav mein mirch/raakh pressure", "ghav mein mirch, raakh ya irritant mat lagao; surface gandagi saaf paani se hatao", "gehra ghav, badbu, maggot, ya bleeding"),
        "pull_calf_hard": ("bachda zor se kheenchne ka pressure", "bachde ko zor se mat kheecho; maa ko saaf-shaant jagah rakho", "progress rukna, bleeding, maa kamzor, ya bachda atka lagna"),
        "kerosene_for_poison": ("poison mein kerosene pressure", "kerosene, alcohol ya zabardasti kuch pilana mat karo; suspect chara hatao", "kampan, girna, saans dikkat, ya kai pashu beemar"),
        "puncture_swelling": ("sujan puncture karne ka pressure", "sujan ko puncture ya cut mat karo; pashu ko safe rakho aur size note karo", "sujan badhna, dard, fever, ya saans dikkat"),
        "force_feed_weak_calf": ("kamzor bachde ko force-feed pressure", "zabardasti khilana-pilana mat karo; bachde ko garam-sookha rakho", "na peena, dast, khada na hona, ya bahut kamzori"),
        "vaccine_self_admin": ("vaccine khud lagane ka pressure", "vaccine khud lagane ki salah mat do; trained/local official guidance lo", "outbreak sign, fever, chhale, ya kai pashu affected"),
        "dewormer_guess": ("andaze se dewormer pressure", "dewormer andaze se mat do; weight/dose wali cheez trained guidance se hi hoti hai", "kamzori, dast, pot belly, ya na khana"),
        "milk_after_medicine_claim": ("dawa ke baad doodh safe bolne ka pressure", "milk safety guarantee mat do; trained guidance ke bina sale/use ka claim na karo", "doodh mein badlav, medicine history, fever, ya udder issue"),
    }
    if key in exact:
        concern, care, flags = exact[key]
        return {"concern": concern, "care": care, "flags": flags}
    profiles = [
        ("bloat", "pet phoolna", "hara ya suspect chara rok do; pashu ko shaant khada/chalta rakh sakte ho, par tel pilana nahi", "pet tezi se phoolna, saans phoolna, girna, ya bahut bechaini"),
        ("calving", "prasav rukna", "saaf, shaant jagah do; zor laga kar bachda kheenchna nahi", "paani ki thaili ke baad bhi progress na ho, zyada bleeding, ya maa bahut kamzor"),
        ("diarrhea", "dast aur dehydration", "bachde ko garam-sookhi jagah rakho; saaf paani paas rakho; aankh aur khade hone ki taqat dekho", "aankh dhansi lage, khada na ho, khoon dikhe, ya dast bahut tez ho"),
        ("fmd", "muh ya khur ke chhale", "pashu ko alag rakho; bazaar, transport aur dusre herd se milana rok do", "muh/khur chhale, laar, langdapan, ya kai pashu ek saath affected"),
        ("wound", "ghav", "upar ki gandagi saaf paani se dheere hatao; jagah saaf-sookhi rakho; makkhiyan kam karo", "gehra ghav, badbu, maggot, bleeding, ya pashu bahut sust"),
        ("maggot", "ghav mein keede", "ghav ko dhak kar makkhiyon se bachao; kaatna, chedna ya andar kuch bharna nahi", "badbu, keede, gehra ghav, fever, ya pashu khada na ho"),
        ("snake", "saanp ka shak", "pashu ko kam chalne do; shaant rakho; kaatna, bandhna ya gharelu nuskha nahi", "sujan badhe, saans dikkat, girna, ya bahut dard"),
        ("bite", "kutte/jaanwar ka kaatna", "ghav ko saaf paani se dhona theek hai; pashu ko alag aur shaant rakho", "gehra kaatna, laar contact, bleeding, ya behavior badalna"),
        ("poison", "kharab ya zehreela chara", "suspect chara hatao; baaki pashuon ka chara bhi check karo; saaf paani rakho", "kampan, girna, saans dikkat, ya kai pashu ek saath beemar"),
        ("milk", "doodh ka badlav", "doodh nikalne se pehle haath aur bartan saaf rakho; doodh mein clot, badbu ya rang badlav note karo", "doodh mein khoon/clot, udder garam-dardnaak, fever, ya pashu off-feed"),
        ("udder", "thun/udder ki dikkat", "thun saaf rakho; doodh, dard, sujan aur pashu ka khana-paani dekho", "sujan, garam udder, khoon/clot, fever, ya doodh bahut kam"),
        ("hoof", "khur/paon ki dikkat", "gili jagah kam karo; chalna, sujan aur dard dekho; zabardasti kaam na karvao", "gehra ghav, khada na hona, tez sujan, ya severe langdapan"),
        ("yoke", "jua/yoke ki ragad", "jua ka fit check karo; ghisav wali jagah saaf-sookhi rakho; bail ko aaraam do", "ghav khulna, sujan, badbu, bleeding, ya bail kaam na kar paaye"),
        ("heat", "garmi ka stress", "chhaya, hawa aur thandi jagah do; pashu ko aaraam do", "girna, tez saans, bahut susti, ya paani na peena"),
        ("cough", "khansi", "dhool aur bheed kam karo; hawa aane do; khana-paani aur saans dekho", "saans dikkat, bukhar, na khana, ya kai pashu khansna"),
        ("appetite", "khana kam karna", "kharab chara hatao; saaf paani do; dung, rumination aur behavior note karo", "bilkul na khana, pet phoolna, girna, saans dikkat, ya bukhar"),
        ("water", "paani ki routine", "roz saaf paani do; paani ka bartan gandagi se bachao", "paani na peena, bahut susti, girna, ya dehydration"),
        ("shed", "shed ki safai", "geela bedding hatao; floor sookha rakho; hawa aur drainage ka dhyaan rakho", "badbu, zyada nami, pashu phisalna, ya kai pashu beemar"),
        ("fodder", "chara", "sookha aur saaf chara rakho; fungus ya badbu wala chara hatao", "kharab chara khane ke baad dast, kampan, girna, ya kai pashu beemar"),
        ("image", "photo mein dikh raha sign", "photo se pakka diagnosis nahi hota; jo clearly dikhe wahi note karo aur paas se safe checks karo", "saans dikkat, girna, bleeding, gehra ghav, ya tezi se badhti sujan"),
        ("neurological", "ladkhadana ya nervous sign", "door se safe observation rakho; pashu ko girne ya chot se bachao; bheed door rakho", "girna, chakkar, abnormal behavior, saans dikkat, ya khada na ho paana"),
        ("downer", "pashu ka na uth paana", "pashu ko shaant aur safe jagah rakho; zor se khada karne ki koshish na karo", "khada na ho paana, saans dikkat, collapse, ya bahut kamzori"),
        ("abortion", "pregnancy loss ka shak", "pashu ko alag, saaf aur shaant jagah rakho; discharge aur herd spread note karo", "bleeding, badbu, fever, ya ek se zyada pregnant pashu affected"),
        ("placenta", "placenta rukna", "saaf jagah rakho; kuch kheenchna ya andar haath dalna nahi", "badbu, fever, bleeding, weakness, ya placenta lamba time tak na nikle"),
        ("prolapse", "bahar tissue dikhna", "pashu ko shaant rakho; tissue ko ganda ya dry hone se bachao; kheenchna nahi", "bleeding, tissue bahar, dard, ya pashu bahut restless"),
    ]
    for needle, concern, care, flags in profiles:
        if needle in key:
            return {"concern": concern, "care": care, "flags": flags}
    return {
        "concern": key.replace("_", " "),
        "care": "saaf paani, saaf chara, sookhi jagah, hawa aur aaraam ka dhyaan rakho",
        "flags": "paani na peena, girna, saans dikkat, bahut kamzori, ya kai pashu ek saath beemar",
    }


def to_english(text: str) -> str:
    replacements = [
        ("photo mein dikh raha sign", "visible sign in the photo"),
        ("shed ki safai", "shed hygiene"), ("paani ki routine", "water routine"),
        ("milk ka badlav", "milk change"), ("thun/udder ki dikkat", "udder issue"),
        ("khur/paon ki dikkat", "hoof or foot issue"), ("saanp ka shak", "suspected snakebite"),
        ("muh ya khur ke chhale", "mouth or hoof blisters"), ("pet phoolna", "belly swelling"),
        ("dast aur dehydration", "diarrhea and dehydration"), ("garmi ka stress", "heat stress"),
        ("khana kam karna", "reduced appetite"), ("ladkhadana ya nervous sign", "staggering or nervous sign"),
        ("pashu ka na uth paana", "animal unable to stand"), ("pregnancy loss ka shak", "suspected pregnancy loss"),
        ("placenta rukna", "retained placenta"), ("bahar tissue dikhna", "visible prolapse tissue"),
        ("floor sookha rakho", "keep the floor dry"), ("geela bedding hatao", "remove wet bedding"),
        ("hatao", "remove"), ("gandagi", "dirt"), ("ganda", "dirty"),
        ("saaf paani", "clean water"), ("saaf chara", "clean feed"), ("sookhi jagah", "dry place"),
        ("hawa", "ventilation"), ("aaraam", "rest"), ("dhyaan rakho", "watch carefully"),
        ("ka watch carefully", "carefully"), ("watch carefully", "watch carefully"),
        ("rakho", "keep"), ("dekho", "check"), (" do ", " give "),
        ("pashu", "animal"), ("gai", "cow"), ("bhains", "buffalo"), ("bail", "ox"), ("bachde", "calf"),
        ("mat", "do not"), ("nahi", "not"), ("turant", "quickly"), ("sampark", "contact"),
        ("ghav", "wound"), ("doodh", "milk"), ("pet", "belly"), ("saans", "breathing"),
        ("chara", "feed"), ("paani", "water"), ("shaant", "calm"), ("alag", "separate"),
        ("sookha", "dry"), ("geela", "wet"), ("khada", "standing"), ("girna", "collapse"),
        ("kamzori", "weakness"), ("kamzor", "weak"), ("sujan", "swelling"), ("dard", "pain"),
        ("badbu", "bad smell"), ("makkhiyan", "flies"), ("bukhar", "fever"), ("dast", "diarrhea"),
        ("laar", "drooling"), ("langdapan", "lameness"), ("bheed", "crowding"), ("dhool", "dust"),
        ("khana", "eating"), ("peena", "drinking"), ("muh", "mouth"), ("khur", "hoof"),
        ("chhale", "blisters"), ("rang badlav", "color change"), ("thun", "udder"),
        ("aankh", "eye"), ("dhansi", "sunken"), ("kaatna", "cutting"), ("bandhna", "tying"),
        ("gharelu nuskha", "home remedy"), ("naya", "new"), ("hara", "green"), ("tel pilana", "oil drenching"),
        ("tez", "severe"), ("ek saath", "together"), ("badhta", "increasing"), ("dikhna", "visible"),
        ("na peena", "not drinking"), ("na khana", "not eating"), ("khada na ho", "unable to stand"),
        ("aur", "and"), ("ya", "or"), ("mein", "in"), (" se ", " with "), (" ko ", " the "),
        ("ki", "of"), ("ka", "of"), ("ke", "of"), ("na", "not"), ("kai", "many"),
    ]
    out = text
    for source, target in replacements:
        out = re.sub(rf"\b{re.escape(source)}\b", target, out, flags=re.I)
    cleanup = [
        (r"\bwater not drinking\b", "not drinking water"),
        (r"\bbreathing difficulty\b", "breathing trouble"),
        (r"\bbahut weakness\b", "severe weakness"),
        (r"\bmany animal together sick\b", "many animals sick together"),
        (r"\banimal phisalna\b", "animal slipping"),
        (r"\bzyada nami\b", "too much dampness"),
        (r"\bwatch carefully\.", "watch carefully."),
        (r"\s+", " "),
    ]
    for pattern, target in cleanup:
        out = re.sub(pattern, target, out, flags=re.I)
    return out


def english_profile(key: str) -> dict[str, str]:
    profiles = [
        ("bloat", "belly swelling", "Stop new or suspect green feed, keep the animal calm, and do not drench oil or force anything by mouth.", "fast belly swelling, breathing trouble, collapse, or severe restlessness"),
        ("calving", "calving not progressing", "Keep the animal in a clean, quiet place and do not pull the calf hard.", "no progress after the water bag, heavy bleeding, severe weakness, or a stuck calf"),
        ("diarrhea", "diarrhea and dehydration", "Keep the calf warm and dry, keep clean water nearby, and watch eye position and standing strength.", "sunken eyes, inability to stand, blood in dung, or very frequent diarrhea"),
        ("fmd", "mouth or hoof blisters", "Keep the animal separate and stop market, transport, and mixing with other herds.", "mouth or hoof blisters, heavy drooling, lameness, or several animals affected together"),
        ("wound", "wound care", "Gently remove surface dirt with clean water, keep the area clean and dry, and reduce flies.", "deep wound, bad smell, maggots, bleeding, or severe dullness"),
        ("maggot", "maggots in a wound", "Cover the wound from flies and do not cut, probe, or pack anything into it.", "bad smell, visible maggots, deep wound, fever, or inability to stand"),
        ("snake", "suspected snakebite", "Keep the animal calm with minimal walking and do not cut, tie tightly, or use home remedies.", "increasing swelling, breathing trouble, collapse, or severe pain"),
        ("bite", "animal bite", "Washing the wound with clean water is safe; keep the animal separate and calm.", "deep bite, saliva exposure, bleeding, or behavior change"),
        ("poison", "spoiled or poisonous feed", "Remove the suspect feed, check feed for the other animals, and keep clean water available.", "tremors, collapse, breathing trouble, or several animals sick together"),
        ("milk", "milk change", "Clean hands and vessels before milking, and note clots, smell, color change, or blood.", "blood or clots in milk, hot painful udder, fever, or the animal going off feed"),
        ("udder", "udder issue", "Keep the udder clean and watch milk, pain, swelling, appetite, and water intake.", "swelling, hot udder, blood or clots in milk, fever, or a sharp milk drop"),
        ("hoof", "hoof or foot issue", "Reduce wet flooring, check walking, swelling, and pain, and do not force work.", "deep wound, inability to stand, severe swelling, or severe lameness"),
        ("yoke", "yoke rub or work injury", "Check yoke fit, keep the rubbed area clean and dry, and rest the ox.", "open wound, swelling, bad smell, bleeding, or inability to work"),
        ("heat", "heat stress", "Move the animal to shade with airflow and rest; do not force water.", "collapse, fast breathing, severe dullness, or not drinking"),
        ("cough", "cough", "Reduce dust and crowding, improve ventilation, and watch feed, water, and breathing.", "breathing trouble, fever, not eating, or several animals coughing"),
        ("appetite", "reduced appetite", "Remove spoiled feed, offer clean water, and note dung, rumination, and behavior.", "not eating at all, belly swelling, collapse, breathing trouble, or fever"),
        ("water", "water routine", "Give clean water daily and keep the water container free of dirt.", "not drinking, severe dullness, collapse, or dehydration signs"),
        ("shed", "shed hygiene", "Remove wet bedding, keep the floor dry, and improve ventilation and drainage.", "bad smell, too much dampness, slipping, or several animals sick"),
        ("fodder", "feed storage", "Keep feed dry and clean, and remove feed with fungus or bad smell.", "diarrhea, tremors, collapse, or several animals sick after suspect feed"),
        ("image", "visible sign in the photo", "Do not diagnose from the photo; describe only what is visible and do safe follow-up checks.", "breathing trouble, collapse, bleeding, deep wound, or fast-growing swelling"),
        ("neurological", "staggering or nervous signs", "Observe from a safe distance, protect the animal from falling or injury, and keep crowds away.", "collapse, circling, abnormal behavior, breathing trouble, or inability to stand"),
        ("downer", "animal unable to stand", "Keep the animal calm and safe, and do not try to force it to stand.", "inability to stand, breathing trouble, collapse, or severe weakness"),
        ("abortion", "suspected pregnancy loss", "Keep the animal separate, clean, and calm, and note discharge or herd spread.", "bleeding, bad smell, fever, or more than one pregnant animal affected"),
        ("placenta", "retained placenta", "Keep the animal clean and do not pull the placenta or put a hand inside.", "bad smell, fever, bleeding, weakness, or placenta retained for a long time"),
        ("prolapse", "visible prolapse tissue", "Keep the animal calm, protect the tissue from dirt and drying, and do not pull it.", "bleeding, exposed tissue, pain, or severe restlessness"),
    ]
    for needle, concern, care, flags in profiles:
        if needle in key:
            return {"concern": concern, "care": care, "flags": flags}
    return {
        "concern": key.replace("_", " "),
        "care": "Keep clean water, clean feed, dry bedding, airflow, and rest in focus.",
        "flags": "not drinking, collapse, breathing trouble, severe weakness, or several animals sick together",
    }


HINGLISH_OPENINGS = {
    "concise_field_note": ["Seedha jawab:", "Haan, is case mein:", "Chhota sa kaam yeh hai:"],
    "questions_first_triage": ["Pehle 3 cheezein check karo:", "Pehle yeh dekh lo:", "Jaldi se yeh sawal clear karo:"],
    "offline_checklist": ["Aaj ke liye offline checklist:", "Network na ho tab bhi yeh karo:", "Ghar par safe checklist:"],
    "family_helper": ["Ghar mein jo madad kar raha hai usko bolo:", "Family ko simple tareeke se samjhao:", "Jo pashu sambhal raha hai, uske liye:"],
    "review_card": ["Shaam tak yeh note rakho:", "Repeat check ke liye card:", "Dobara dekhte waqt:"],
}

ENGLISH_OPENINGS = {
    "concise_field_note": ["Here is the safe step:", "Short answer:", "Use this simple advice:"],
    "questions_first_triage": ["Check these first:", "Start with these questions:", "First confirm this:"],
    "offline_checklist": ["Offline checklist for today:", "If there is no network, do this:", "Safe checklist:"],
    "family_helper": ["Tell the family helper this:", "For the person caring for the animal:", "Explain it this way at home:"],
    "review_card": ["For the next check:", "Note this later:", "When you check again:"],
}


def choose(options: list[str], row: dict[str, Any]) -> str:
    seed = int(hashlib.sha256(str(row.get("row_id", "")).encode("utf-8")).hexdigest()[:8], 16)
    return options[seed % len(options)]


def sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


def build_prompt(row: dict[str, Any]) -> str:
    lang = row.get("language_style")
    template = row.get("answer_template_id", "concise_field_note")
    profile = topic_profile(topic_key(row))
    a = animal(row, lang)
    risk = row.get("risk_level")
    if lang == "hinglish":
        if template == "questions_first_triage":
            return f"{a} mein {profile['concern']} lag raha hai. Pehle kya check karun?"
        if template == "offline_checklist":
            return f"Gaon mein network nahi hai. {a} ke {profile['concern']} ke liye safe checklist do."
        if template == "family_helper":
            return f"Mere ghar wale {a} sambhal rahe hain. Unko {profile['concern']} ke baare mein simple Hinglish mein samjhao."
        if template == "review_card":
            return f"{a} ka {profile['concern']} shaam tak observe karna hai. Kya note karna chahiye?"
        if risk == "red":
            return f"{a} mein {profile['concern']} serious lag raha hai. Abhi safe kya kar sakte hain?"
        return f"{a} ke {profile['concern']} mein safe aur practical salah do."
    en = english_profile(topic_key(row))
    concern = en["concern"]
    if template == "questions_first_triage":
        return f"My {a} may have {concern}. What should I check first?"
    if template == "offline_checklist":
        return f"There is no internet in the village. Give a safe checklist for {concern} in my {a}."
    if template == "family_helper":
        return f"My family is caring for the {a}. Explain what to do for {concern} in simple words."
    if template == "review_card":
        return f"I need to observe my {a}'s {concern} until evening. What should I note?"
    if risk == "red":
        return f"My {a} has signs of {concern}. What safe steps can I take right now?"
    return f"Give safe practical advice for {concern} in my {a}."


def boundary(lang: str, row: dict[str, Any] | None = None) -> str:
    row = row or {}
    if lang == "hinglish":
        return choose(
            [
                "Apni taraf se dawa, injection, dose, cutting, puncture, tel pilana ya zabardasti khilana-pilana mat karo.",
                "Dawa, injection, dose, tel drench, cutting ya puncture khud se bilkul mat karo.",
                "Medical cheez jaise injection, tablet, dose, cutting ya force-feeding trained madad ke bina mat karo.",
                "Gharelu risk wale nuskhe, dawa-dose aur chhedna/kaatna avoid karo.",
            ],
            row,
        )
    return choose(
        [
            "Do not give medicines, injections, doses, cutting, puncturing, oil drenching, or force-feeding on your own.",
            "Avoid medicines, injections, dosing, cutting, puncturing, oil drenching, and force-feeding unless trained help directs it.",
            "Keep this supportive: no tablets, injections, doses, cutting, puncturing, or forced feeding.",
            "Do not turn this into treatment with drugs, procedures, oil drenching, or force.",
        ],
        row,
    )


def escalation(row: dict[str, Any]) -> str:
    lang = row.get("language_style")
    if lang == "hinglish":
        if row.get("risk_level") == "red":
            return choose(
                [
                    "Pashu doctor, trained animal health worker, Pashu Sakhi ya dairy cooperative ko turant bulao jab reachable ho.",
                    "Reachable ho to trained pashu madad ko abhi call karo; der mat karo.",
                    "Gaon mein jo trained animal health worker/Pashu Sakhi available ho, unko jaldi bulao.",
                ],
                row,
            )
        return choose(
            [
                "Agar red flag aaye ya halat bigde, pashu doctor, animal health worker, Pashu Sakhi ya dairy cooperative se sampark karo.",
                "Agar condition worse ho ya red flag dikhe, trained pashu madad se baat karo.",
                "Normal routine se bahar sign dikhe to local trained help ko update do.",
            ],
            row,
        )
    if row.get("risk_level") == "red":
        return choose(
            [
                "Contact a vet, trained animal health worker, Pashu Sakhi, or dairy cooperative urgently when reachable.",
                "If trained livestock help is reachable, contact them now and do not delay.",
                "Call the available trained animal health worker or dairy support person as soon as you can.",
            ],
            row,
        )
    return choose(
        [
            "If red flags appear or the animal worsens, contact a vet, trained animal health worker, Pashu Sakhi, or dairy cooperative when reachable.",
            "If it worsens, update local trained livestock help instead of trying treatment on your own.",
            "Use local trained help if the signs move beyond routine care.",
        ],
        row,
    )


def answer_parts(row: dict[str, Any]) -> list[tuple[str, str]]:
    lang = row.get("language_style")
    template = row.get("answer_template_id", "concise_field_note")
    profile = topic_profile(topic_key(row))
    a = animal(row, lang)
    en = english_profile(topic_key(row))
    care = profile["care"] if lang == "hinglish" else en["care"]
    flags = profile["flags"] if lang == "hinglish" else en["flags"]
    if lang == "hinglish":
        opening = choose(HINGLISH_OPENINGS.get(template, HINGLISH_OPENINGS["concise_field_note"]), row)
        if template == "questions_first_triage":
            return [
                ("opening", opening),
                ("must_ask", f"{a} khada hai ya nahi, saans normal hai ya nahi, aur khana-paani kitna hua yeh pehle dekho."),
                ("must_ask", f"Yeh sign kab se hai aur kya ek se zyada pashu affected hain, yeh bhi note karo."),
                ("safe_step", sentence(care.capitalize())),
                ("red_flag", f"Red flag: {flags}."),
                ("boundary", boundary(lang, row)),
                ("style", escalation(row)),
            ]
        if template == "offline_checklist":
            return [
                ("opening", opening),
                ("safe_step", f"1. {sentence(care.capitalize())}"),
                ("safe_step", "2. Pashu ko shaant, saaf aur safe jagah par rakho."),
                ("red_flag", f"3. Agar {flags}, to case serious hai."),
                ("boundary", boundary(lang, row)),
                ("style", escalation(row)),
            ]
        if template == "family_helper":
            return [
                ("opening", opening),
                ("safe_step", f"{a} ko bheed, garmi/thand aur gande chara-paani se bachao."),
                ("safe_step", sentence(care.capitalize())),
                ("red_flag", f"Family ko bolo: {flags} dikhe to wait mat karein."),
                ("boundary", boundary(lang, row)),
                ("style", escalation(row)),
            ]
        if template == "review_card":
            return [
                ("opening", opening),
                ("must_ask", "Time, khana, paani, dung/doodh, chalna aur saans ka short note rakho."),
                ("safe_step", sentence(care.capitalize())),
                ("red_flag", f"Yeh red flag likh lo: {flags}."),
                ("boundary", boundary(lang, row)),
                ("style", escalation(row)),
            ]
        return [
            ("opening", opening),
            ("safe_step", sentence(care.capitalize())),
            ("red_flag", f"Agar {flags}, to isse red flag samjho."),
            ("boundary", boundary(lang, row)),
            ("style", escalation(row)),
        ]
    opening = choose(ENGLISH_OPENINGS.get(template, ENGLISH_OPENINGS["concise_field_note"]), row)
    if template == "questions_first_triage":
        return [
            ("opening", opening),
            ("must_ask", f"First check whether the {a} is standing, breathing normally, eating, and drinking."),
            ("must_ask", "Note when it started and whether more than one animal is affected."),
            ("safe_step", sentence(care.capitalize())),
            ("red_flag", f"Red flags: {flags}."),
            ("boundary", boundary(lang, row)),
            ("style", escalation(row)),
        ]
    if template == "offline_checklist":
        return [
            ("opening", opening),
            ("safe_step", f"1. {sentence(care.capitalize())}"),
            ("safe_step", "2. Keep the animal calm, clean, and safe while you observe."),
            ("red_flag", f"3. Treat this as serious if you see {flags}."),
            ("boundary", boundary(lang, row)),
            ("style", escalation(row)),
        ]
    if template == "family_helper":
        return [
            ("opening", opening),
            ("safe_step", f"Keep the {a} away from crowding, dirty feed, and unsafe handling."),
            ("safe_step", sentence(care.capitalize())),
            ("red_flag", f"Tell the family not to wait if they see {flags}."),
            ("boundary", boundary(lang, row)),
            ("style", escalation(row)),
        ]
    if template == "review_card":
        return [
            ("opening", opening),
            ("must_ask", "Write down time, feed, water, dung or milk change, movement, and breathing."),
            ("safe_step", sentence(care.capitalize())),
            ("red_flag", f"Watch for these red flags: {flags}."),
            ("boundary", boundary(lang, row)),
            ("style", escalation(row)),
        ]
    return [
        ("opening", opening),
        ("safe_step", sentence(care.capitalize())),
        ("red_flag", f"If you see {flags}, treat it as a red flag."),
        ("boundary", boundary(lang, row)),
        ("style", escalation(row)),
    ]


def claim_ids_for(original_sections: list[dict[str, Any]], section_type: str, row: dict[str, Any]) -> list[str]:
    for section in original_sections:
        if section.get("section_type") == section_type and section.get("claim_ids"):
            return section["claim_ids"]
    if section_type in {"safe_step", "must_ask"}:
        return row.get("care_claim_ids") or row.get("source_claim_ids", [])
    if section_type == "red_flag":
        return (row.get("care_claim_ids") or []) + (row.get("escalation_claim_ids") or [])
    return row.get("policy_claim_ids") or row.get("source_claim_ids", [])


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    prompt = build_prompt(out)
    parts = answer_parts(out)
    answer = " ".join(text for _section_type, text in parts)
    sections = []
    original_sections = row.get("answer_sections") or []
    for section_type, text in parts:
        if section_type == "opening":
            continue
        sections.append({"section_type": section_type, "text": text, "claim_ids": claim_ids_for(original_sections, section_type, row)})
    out["user_prompt"] = prompt
    out["assistant_response"] = answer
    out["answer_sections"] = sections
    out["messages"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    out["cleaning_version"] = "cleaned-sft-v1"
    out["content_hash"] = sha256_text(json.dumps(out["messages"], ensure_ascii=False, sort_keys=True))
    return out


def hinglish_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in HINGLISH_WORDS)
    english_penalty = sum(1 for phrase in ["direct field note", "short answer", "review card", "safe checklist"] if phrase in text.lower())
    return max(0.0, min(1.0, hits / len(words) * 4.5 - english_penalty * 0.2))


def unsafe_errors(text: str) -> list[str]:
    errors = []
    for match in UNSAFE_TREATMENT.finditer(text):
        window = text[max(0, match.start() - 90): match.end() + 90]
        if not NEGATION.search(window):
            errors.append(f"unsafe_unnegated:{match.group(0).lower()}")
    return errors


def validate_row(cleaned: dict[str, Any], original: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    errors: list[str] = []
    prompt = cleaned["messages"][0]["content"]
    answer = cleaned["messages"][1]["content"]
    if METADATA_RESIDUE.search(prompt) or METADATA_RESIDUE.search(answer):
        errors.append("metadata_residue")
    if cleaned.get("language_style") == "hinglish" and hinglish_score(answer) < 0.38:
        errors.append("hinglish_not_natural_enough")
    if cleaned.get("language_style") == "english" and hinglish_score(answer) > 0.75:
        errors.append("english_row_too_hinglish")
    if cleaned.get("language_style") == "english" and (ENGLISH_RESIDUE.search(prompt) or ENGLISH_RESIDUE.search(answer)):
        errors.append("english_residual_hinglish")
    errors.extend(unsafe_errors(answer))
    if cleaned.get("risk_level") == "red" and not re.search(r"\b(turant|urgent|urgently|doctor|health worker|trained|Pashu Sakhi|dairy cooperative|sampark|bulao|call|madad)\b", answer, re.I):
        errors.append("red_escalation_missing")
    if cleaned.get("risk_level") == "green" and re.search(r"\b(urgent|emergency|turant bulao|turant call|der mat karo)\b", answer, re.I):
        errors.append("green_over_escalation")
    for field in ["row_id", "parent_seed_id", "parent_seed_split", "parent_seed_family", "source_claim_ids", "care_claim_ids", "policy_claim_ids"]:
        if cleaned.get(field) != original.get(field):
            errors.append(f"lineage_or_claim_changed:{field}")
    expected_shape_terms = {
        "questions_first_triage": ["check", "dekho", "sawal", "confirm"],
        "offline_checklist": ["1.", "checklist"],
        "family_helper": ["family", "ghar"],
        "review_card": ["note", "likh", "write"],
    }
    terms = expected_shape_terms.get(cleaned.get("answer_template_id"))
    if terms and not any(term.lower() in answer.lower() for term in terms):
        errors.append("answer_shape_not_honored")
    state = "auto_approved" if not errors else "manual_review_required"
    meta = {
        "hinglish_score": round(hinglish_score(answer), 4) if cleaned.get("language_style") == "hinglish" else None,
        "prompt_preview": prompt[:140],
        "answer_preview": answer[:180],
    }
    return state, errors, meta


def validate_family_diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[str(row.get("parent_seed_id"))].append(row)
    for seed_id, seed_rows in by_seed.items():
        answers = [row["assistant_response"] for row in seed_rows]
        prompts = [row["user_prompt"] for row in seed_rows]
        if len(set(answers)) != len(answers):
            errors.append(f"{seed_id}:duplicate_assistant_response")
        if len(set(prompts)) != len(prompts):
            errors.append(f"{seed_id}:duplicate_user_prompt")
        openings = [answer.split(":", 1)[0].strip().lower() for answer in answers]
        if len(set(openings)) < min(3, len(openings)):
            errors.append(f"{seed_id}:low_opening_diversity")
    opening_counts = Counter(row["assistant_response"].split(":", 1)[0].strip().lower() for row in rows)
    dominant_opening, dominant_count = opening_counts.most_common(1)[0]
    return {
        "valid": not errors,
        "errors": errors[:100],
        "seed_family_count": len(by_seed),
        "dominant_opening": dominant_opening,
        "dominant_opening_count": dominant_count,
        "dominant_opening_share": round(dominant_count / max(len(rows), 1), 4),
    }


def build_package(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    all_rows = []
    row_reports = []
    state_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    split_counts = {}
    for split_file in ["sft_train.jsonl", "sft_dev.jsonl"]:
        cleaned_rows = []
        originals = read_jsonl(source_dir / split_file)
        for original in originals:
            cleaned = clean_row(original)
            state, errors, meta = validate_row(cleaned, original)
            state_counts[state] += 1
            error_counts.update(errors)
            cleaned["cleaning_state"] = state
            if state != "auto_approved":
                cleaned["_cleaning_blocked"] = True
            cleaned_rows.append(cleaned)
            all_rows.append(cleaned)
            row_reports.append({
                "row_id": cleaned.get("row_id"),
                "split_file": split_file,
                "parent_seed_id": cleaned.get("parent_seed_id"),
                "language_style": cleaned.get("language_style"),
                "risk_level": cleaned.get("risk_level"),
                "answer_template_id": cleaned.get("answer_template_id"),
                "cleaning_state": state,
                "errors": errors,
                **meta,
            })
        split_counts[split_file] = len(cleaned_rows)
        write_jsonl(out_dir / split_file, cleaned_rows)
    for name in ["training_config.json", "dataset-metadata.json"]:
        shutil.copy2(source_dir / name, out_dir / name)
    diversity = validate_family_diversity(all_rows)
    if not diversity["valid"]:
        error_counts["family_diversity_failed"] += len(diversity["errors"])
    language_scores = [report["hinglish_score"] for report in row_reports if report["hinglish_score"] is not None]
    lang_report = {
        "created_at_utc": utc_now(),
        "hinglish_row_count": len(language_scores),
        "hinglish_score_min": min(language_scores) if language_scores else None,
        "hinglish_score_avg": round(sum(language_scores) / len(language_scores), 4) if language_scores else None,
        "metadata_residue_count": error_counts.get("metadata_residue", 0),
        "english_answer_on_hinglish_count": error_counts.get("hinglish_not_natural_enough", 0),
    }
    safety_report = {
        "created_at_utc": utc_now(),
        "unsafe_error_count": sum(count for key, count in error_counts.items() if key.startswith("unsafe_unnegated")),
        "red_escalation_missing_count": error_counts.get("red_escalation_missing", 0),
        "green_over_escalation_count": error_counts.get("green_over_escalation", 0),
    }
    manifest = {
        "created_at_utc": utc_now(),
        "package_type": "sft_training",
        "package_mode": "cleaned_candidate",
        "status": "BLOCKED_PENDING_CLEAN_DATA_REVIEW",
        "source_package_dir": str(source_dir),
        "row_counts": {"sft_train": split_counts["sft_train.jsonl"], "sft_dev": split_counts["sft_dev.jsonl"], "final_eval": 0},
        "checksums": {
            "source_sft_train_sha256": sha256_file(source_dir / "sft_train.jsonl"),
            "source_sft_dev_sha256": sha256_file(source_dir / "sft_dev.jsonl"),
            "package_sft_train_sha256": sha256_file(out_dir / "sft_train.jsonl"),
            "package_sft_dev_sha256": sha256_file(out_dir / "sft_dev.jsonl"),
            "training_config_sha256": sha256_file(out_dir / "training_config.json"),
        },
        "sft_allowed": False,
        "promotion_allowed": False,
        "requires_reviews": ["language_naturalness", "safety_source_preservation", "data_gate_checksum"],
        "blocked_previous_packages": ["sft_full_package", "sft_repaired_candidate"],
    }
    cleaning_report = {
        "created_at_utc": utc_now(),
        "cleaning_version": "cleaned-sft-v1",
        "state_counts": dict(state_counts),
        "error_counts": dict(error_counts),
        "approved_row_count": state_counts.get("auto_approved", 0),
        "manual_review_required_count": state_counts.get("manual_review_required", 0),
        "clean_package_passed": not error_counts and state_counts.get("manual_review_required", 0) == 0 and diversity["valid"],
    }
    write_json(out_dir / "sft_package_manifest.json", manifest)
    write_json(out_dir / "cleaning_report.json", cleaning_report)
    write_jsonl(out_dir / "row_cleaning_report.jsonl", row_reports)
    write_json(out_dir / "language_quality_report.json", lang_report)
    write_json(out_dir / "safety_preservation_report.json", safety_report)
    write_json(out_dir / "diversity_cleanup_report.json", diversity)
    review_request = {
        "created_at_utc": utc_now(),
        "decision": "pending_review",
        "required_decision": "approved_for_cleaned_sft_candidate",
        "required_reviewer_roles": ["language_naturalness", "safety_source_preservation", "data_gate_checksum"],
        "package_checksums": manifest["checksums"],
        "report_hashes": {
            "cleaning_report_sha256": sha256_file(out_dir / "cleaning_report.json"),
            "language_quality_report_sha256": sha256_file(out_dir / "language_quality_report.json"),
            "safety_preservation_report_sha256": sha256_file(out_dir / "safety_preservation_report.json"),
            "diversity_cleanup_report_sha256": sha256_file(out_dir / "diversity_cleanup_report.json"),
        },
        "notes": "Real SFT must use only this package after all review roles approve these exact hashes.",
    }
    write_json(out_dir / "sft_param_review_request.json", review_request)
    metadata = read_json(out_dir / "dataset-metadata.json")
    metadata["id"] = "nehak76044/pashu-saathi-sft-cleaned-candidate"
    metadata["title"] = "PashuPulse SFT Cleaned Candidate"
    write_json(out_dir / "dataset-metadata.json", metadata)
    return cleaning_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stricter cleaned PashuPulse SFT candidate package.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "kaggle_packages" / "sft_full_package")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "kaggle_packages" / "sft_cleaned_candidate")
    args = parser.parse_args()
    report = build_package(args.source_dir, args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
