import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_special_tokens(raw_text: str) -> List[str]:
    tokens = []

    if any(word.isupper() and len(word) > 1 for word in re.findall(r"\b[\w']+\b", raw_text)):
        tokens.append("<CAPS>")

    if raw_text.isupper() and len(raw_text) > 1:
        tokens.append("<ALL_CAPS>")

    if re.search(r"(.)\1{2,}", raw_text):
        tokens.append("<ELONGATED>")

    exclam = raw_text.count("!")
    if exclam == 1:
        tokens.append("<EXCLAM_1>")
    elif exclam == 2:
        tokens.append("<EXCLAM_2>")
    elif exclam >= 3:
        tokens.append("<EXCLAM_3>")

    ques = raw_text.count("?")
    if ques == 1:
        tokens.append("<QUESTION_1>")
    elif ques == 2:
        tokens.append("<QUESTION_2>")
    elif ques >= 3:
        tokens.append("<QUESTION_3>")

    lowered = raw_text.lower()

    positive_words = {
        "love", "great", "good", "amazing", "awesome", "happy",
        "excellent", "nice", "wonderful", "fantastic", "best"
    }
    negative_words = {
        "hate", "bad", "terrible", "awful", "sad", "angry",
        "disappointing", "frustrating", "worst", "horrible"
    }

    if any(w in lowered for w in positive_words):
        tokens.append("<SENT_POS_HINT>")

    if any(w in lowered for w in negative_words):
        tokens.append("<SENT_NEG_HINT>")

    if any(w in lowered for w in ["please", "kindly", "regards", "dear", "sincerely"]):
        tokens.append("<TONE_FORMAL>")
    elif any(w in lowered for w in ["lol", "omg", "bro", "dude", "hey", "nah"]):
        tokens.append("<TONE_CASUAL>")
    else:
        tokens.append("<NEUTRAL_STYLE>")

    if any(w in lowered for w in ["happy", "joy", "glad", "excited", "delighted"]):
        tokens.append("<EMO_HAPPY>")
    elif any(w in lowered for w in ["sad", "unhappy", "depressed", "down"]):
        tokens.append("<EMO_SAD>")
    elif any(w in lowered for w in ["angry", "mad", "furious", "rage"]):
        tokens.append("<EMO_ANGRY>")

    return tokens


def build_model_text(text: str) -> Tuple[str, List[str], str]:
    raw = text.strip()
    cleaned = clean_text(raw)
    tokens = extract_special_tokens(raw)
    model_text = " ".join(tokens + [cleaned]) if tokens else cleaned
    return model_text, tokens, cleaned


def build_explanation(raw_text: str, model_text: str = None) -> List[str]:
    cues = []
    lowered = raw_text.lower()

    if raw_text.count("!") > 0:
        cues.append(f"exclamation marks × {raw_text.count('!')}")

    if raw_text.count("?") > 0:
        cues.append(f"question marks × {raw_text.count('?')}")

    if raw_text.isupper() and len(raw_text) > 1:
        cues.append("all-caps emphasis")

    if re.search(r"(.)\1{2,}", raw_text):
        cues.append("elongated letters")

    cue_map = {
    # ── POSITIVE ──────────────────────────────────────────
    "love": "positive cue",
    "loved": "positive cue",
    "lovely": "positive cue",
    "great": "positive cue",
    "amazing": "positive cue",
    "good": "positive cue",
    "wonderful": "positive cue",
    "fantastic": "positive cue",
    "excellent": "positive cue",
    "brilliant": "positive cue",
    "superb": "positive cue",
    "perfect": "positive cue",
    "happy": "positive cue",
    "happiness": "positive cue",
    "joyful": "positive cue",
    "joy": "positive cue",
    "glad": "positive cue",
    "grateful": "positive cue",
    "thankful": "positive cue",
    "blessed": "positive cue",
    "excited": "positive cue",
    "thrilled": "positive cue",
    "delighted": "positive cue",
    "pleased": "positive cue",
    "cheerful": "positive cue",
    "optimistic": "positive cue",
    "proud": "positive cue",
    "incredible": "positive cue",
    "outstanding": "positive cue",
    "impressive": "positive cue",
    "nice": "positive cue",
    "awesome": "positive cue",
    "yay": "positive cue",
    "hooray": "positive cue",
    "yey": "positive cue",

    # ── NEGATIVE ──────────────────────────────────────────
    "bad": "negative cue",
    "terrible": "negative cue",
    "awful": "negative cue",
    "horrible": "negative cue",
    "dreadful": "negative cue",
    "disgusting": "negative cue",
    "pathetic": "negative cue",
    "unacceptable": "negative cue",
    "worst": "negative cue",
    "poor": "negative cue",
    "useless": "negative cue",
    "failure": "negative cue",
    "failed": "negative cue",
    "fail": "negative cue",
    "disappointing": "negative cue",
    "disappointed": "negative cue",
    "disappointment": "negative cue",
    "hate": "negative cue",
    "hated": "negative cue",
    "hateful": "negative cue",
    "dislike": "negative cue",
    "miserable": "negative cue",
    "unhappy": "negative cue",
    "regret": "negative cue",
    "regretful": "negative cue",
    "unfortunate": "negative cue",
    "unfortunately": "negative cue",
    "unfair": "negative cue",
    "wrong": "negative cue",

    # ── SADNESS ───────────────────────────────────────────
    "sad": "sadness cue",
    "sadness": "sadness cue",
    "sorry": "sadness cue",          # ← fixes "I am sorry"
    "apology": "sadness cue",
    "apologies": "sadness cue",
    "apologize": "sadness cue",
    "apologise": "sadness cue",
    "grief": "sadness cue",
    "grieve": "sadness cue",
    "grieving": "sadness cue",
    "sorrow": "sadness cue",
    "sorrowful": "sadness cue",
    "cry": "sadness cue",
    "crying": "sadness cue",
    "tears": "sadness cue",
    "heartbroken": "sadness cue",
    "heartbreak": "sadness cue",
    "depressed": "sadness cue",
    "depression": "sadness cue",
    "hopeless": "sadness cue",
    "helpless": "sadness cue",
    "lonely": "sadness cue",
    "loneliness": "sadness cue",
    "alone": "sadness cue",
    "miss": "sadness cue",
    "missing": "sadness cue",
    "lost": "sadness cue",
    "hurt": "sadness cue",
    "pain": "sadness cue",
    "painful": "sadness cue",
    "down": "sadness cue",
    "gloomy": "sadness cue",
    "melancholy": "sadness cue",
    "unwanted": "sadness cue",

    # ── ANGER ─────────────────────────────────────────────
    "angry": "anger cue",
    "anger": "anger cue",
    "mad": "anger cue",
    "furious": "anger cue",
    "fury": "anger cue",
    "rage": "anger cue",
    "outraged": "anger cue",
    "outrage": "anger cue",
    "irritated": "anger cue",
    "irritating": "anger cue",
    "annoyed": "anger cue",
    "annoying": "anger cue",
    "infuriated": "anger cue",
    "livid": "anger cue",
    "enraged": "anger cue",
    "bitter": "anger cue",
    "bitterness": "anger cue",
    "hostile": "anger cue",
    "hostility": "anger cue",
    "resentful": "anger cue",
    "resentment": "anger cue",
    "aggressive": "anger cue",
    "aggression": "anger cue",

    # ── STRESS / ANXIETY ──────────────────────────────────
    "frustrating": "stress cue",
    "frustrated": "stress cue",
    "frustration": "stress cue",
    "stressed": "stress cue",
    "stress": "stress cue",
    "stressful": "stress cue",
    "overwhelmed": "stress cue",
    "overwhelming": "stress cue",
    "anxious": "stress cue",
    "anxiety": "stress cue",
    "nervous": "stress cue",
    "nervous wreck": "stress cue",
    "panic": "stress cue",
    "panicking": "stress cue",
    "worried": "stress cue",
    "worry": "stress cue",
    "tense": "stress cue",
    "tension": "stress cue",
    "uneasy": "stress cue",
    "exhausted": "stress cue",
    "burnout": "stress cue",
    "burned out": "stress cue",
    "drained": "stress cue",
    "tired": "stress cue",
    "pressure": "stress cue",

    # ── SURPRISE ──────────────────────────────────────────
    "surprised": "surprise cue",
    "surprise": "surprise cue",
    "shocking": "surprise cue",
    "shocked": "surprise cue",
    "unexpected": "surprise cue",
    "unbelievable": "surprise cue",
    "unbelievably": "surprise cue",
    "astonished": "surprise cue",
    "astonishing": "surprise cue",
    "astounded": "surprise cue",
    "stunned": "surprise cue",
    "jaw-dropping": "surprise cue",
    "whoa": "surprise cue",
    "wow": "surprise cue",
    "omg": "surprise cue",
    "no way": "surprise cue",
    "seriously": "surprise cue",
    "really": "surprise cue",

    # ── FORMAL TONE ───────────────────────────────────────
    "please": "formal cue",
    "kindly": "formal cue",
    "regards": "formal cue",
    "sincerely": "formal cue",
    "dear": "formal cue",
    "respectfully": "formal cue",
    "hereby": "formal cue",
    "pursuant": "formal cue",
    "accordingly": "formal cue",
    "therefore": "formal cue",
    "hence": "formal cue",
    "furthermore": "formal cue",
    "moreover": "formal cue",
    "nevertheless": "formal cue",
    "notwithstanding": "formal cue",
    "attached": "formal cue",
    "enclosed": "formal cue",
    "greetings": "formal cue",
    "salutation": "formal cue",
    "request": "formal cue",
    "inform": "formal cue",
    "acknowledge": "formal cue",

    # ── CASUAL TONE ───────────────────────────────────────
    "lol": "casual cue",
    "lmao": "casual cue",
    "lmfao": "casual cue",
    "rofl": "casual cue",
    "haha": "casual cue",
    "hehe": "casual cue",
    "bro": "casual cue",
    "dude": "casual cue",
    "man": "casual cue",
    "hey": "casual cue",
    "nah": "casual cue",
    "yeah": "casual cue",
    "yep": "casual cue",
    "nope": "casual cue",
    "gonna": "casual cue",
    "wanna": "casual cue",
    "gotta": "casual cue",
    "kinda": "casual cue",
    "sorta": "casual cue",
    "dunno": "casual cue",
    "tbh": "casual cue",
    "imo": "casual cue",
    "idk": "casual cue",
    "btw": "casual cue",
    "fyi": "casual cue",
    "ikr": "casual cue",
    "irl": "casual cue",
    "fr": "casual cue",
    "ngl": "casual cue",
    "lit": "casual cue",
    "vibe": "casual cue",
    "vibes": "casual cue",
    "chill": "casual cue",
    "legit": "casual cue",
    "totally": "casual cue",
    "basically": "casual cue",

    # ── EMOJI TO EMOTION TONE ───────────────────────────────────────
    "😊": "happy",
    "😍": "love",
    "😡": "anger",
    "😡": "anger",
    "🤬": "anger",
    "😒": "sad",
    "😭": "sad",
    "😒": "sad",
    "😂": "joy",
    "🤣": "joy",
    "😘": "love",
    "😚": "love",
    "😇": "love",
    "🥰": "love",
    "😜": "joy",
    "🤪": "joy",
    "😛": "joy",
    "🤑": "joy",
    "🤔": "surprise",
    "😳": "surprise",
    "😱": "surprise",
    "🤯": "surprise",
    "🤨": "surprise"
    }

    for word, label in cue_map.items():
        if word in lowered:
            cues.append(f"{word} → {label}")

    if model_text:
        special_tokens = re.findall(r"<[^>]+>", model_text)
        for tok in special_tokens:
            cues.append(f"{tok} token added")

    seen = set()
    output = []
    for c in cues:
        if c not in seen:
            output.append(c)
            seen.add(c)

    return output[:6]