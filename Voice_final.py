"""
Voice → Keyword → Unreal Bridge (Deepgram + OpenAI)

Final pipeline (deterministic + LLM-assisted):
0) Unit codes: "w a one zero" → WA10. If in fuzzy → RETURN.
0.3) LLM floor gate: "second the floor", "3rd level" → "2"/"3" (must be in fuzzy) → RETURN.
0.5) CloserLook gate (strict cues incl. "in detail(s)"): room/unit in utterance → CloserLook{That}; else → CloserLook{Last}. RETURN if in fuzzy.
1) Exact token whole-word in fuzzy → RETURN.
2) Exact sentence (case/punct-insensitive) in sentences file → return its #Keyword **only if** that keyword is in fuzzy → RETURN.
2.5) Alias layer from LLM (seeded by your sentences): literal phrase or semantic match → RETURN.
3) Router (few-shot using nearest examples from your sentences) → ONE fuzzy token or DROP.
4) Time/Weather-only token-sim fallback (safe) → RETURN.
5) Guards: "apartment/unit/flat" with no code → DROP; "floor/level…" with no number/ordinal (after floor gate) → DROP; else DROP.

Sends exactly ONE token (from fuzzy) to Unreal via WebSocket on change.
"""

# =========================
#           Imports
# =========================

import os
import re
import time
import json
import asyncio
from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

import numpy as np
import sounddevice as sd
import websockets
import aiohttp

from openai import OpenAI
from deepgram import Deepgram


# =========================
#         Config
# =========================

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",   "")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("Set DEEPGRAM_API_KEY in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Audio
SAMPLE_RATE    = 16000
CHUNK_DURATION = 0.5
CHUNK_SIZE     = int(SAMPLE_RATE * CHUNK_DURATION)

# Debounce
DEBOUNCE_SEC      = 1.0
MERGE_WINDOW_SEC  = 2.3

# Thresholds
CONF_THRESHOLD           = 0.75
ALIAS_PHRASE_THRESHOLD   = 0.80  # utterance ↔ alias phrase
SIM_TW_THRESHOLD         = 0.76  # token-sim, but only for time/weather/map class

# Runtime state
detected_text        = "Unknown"
last_place_keyword   = None    # last room OR unit (Kitchen, Bathroom, WA10, WB4, ...)
connected_clients    = set()
buffer_text          = ""
last_transcript_time = 0.0
debounce_task        = None


# =========================
#    Helpers / File IO
# =========================

def clean_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"[^\w\s]", "", s.lower().strip())

def tokens_of(s: str) -> List[str]:
    return clean_text(s).split()

def phrase_in_tokens(tokens: List[str], phrase: str) -> bool:
    """Contiguous phrase match against token list."""
    p = tokens_of(phrase)
    if not p:
        return False
    if len(p) == 1:
        return p[0] in tokens
    n, m = len(tokens), len(p)
    for i in range(0, n - m + 1):
        if tokens[i:i+m] == p:
            return True
    return False

def path_or_default(pref: str, fallback: str) -> str:
    return pref if os.path.exists(pref) else fallback

FUZZY_PATH = path_or_default("fuzzy_keywords.txt", "fuzzy_keywords.txt")
SENT_PATH  = path_or_default("sentence_to_keyword.txt", "sentence_to_keyword.txt")

def load_sentence_keyword_map(path=SENT_PATH) -> Dict[str, str]:
    """Read '#Keyword' headings with example lines under each; map cleaned sentence → Keyword."""
    mapping = {}
    current_keyword = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    current_keyword = line[1:].strip()
                    continue
                if current_keyword:
                    mapping[clean_text(line)] = current_keyword
    except Exception as e:
        print("Error loading sentence_to_keyword.txt:", e)
    print(f"[Summary] Loaded {len(mapping)} sentences from {path}")
    return mapping

def load_allowed_keywords(path=FUZZY_PATH) -> List[str]:
    """Read allowed tokens (fuzzy list). De-duplicate, preserve order."""
    allowed = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                allowed.append(line)
    except Exception as e:
        print("Error loading fuzzy_keywords.txt:", e)
    seen=set(); deduped=[]
    for k in allowed:
        if k not in seen:
            deduped.append(k); seen.add(k)
    print(f"[Summary] Loaded {len(deduped)} allowed keywords from {path}")
    return deduped

sentence_map      = load_sentence_keyword_map()
allowed_keywords  = load_allowed_keywords()
allowed_set       = set(allowed_keywords)
allowed_clean_to_orig: Dict[str, str] = {clean_text(k): k for k in allowed_keywords}

# Group examples by keyword (used to seed alias building)
examples_by_kw: DefaultDict[str, List[str]] = defaultdict(list)
for s_clean, kw in sentence_map.items():
    if len(examples_by_kw[kw]) < 6:
        examples_by_kw[kw].append(s_clean)

# Token classes / helpers
def is_floor_token(tok: str) -> bool:
    return tok.isdigit()

def is_unit_token(tok: str) -> bool:
    return bool(re.fullmatch(r"W[A-B]?\d+", tok))

def is_closerlook(tok: str) -> bool:
    return tok.startswith("CloserLook")

MAIN_ROOMS = {"Kitchen", "Master", "Living", "Guest", "Bathroom"}

# Restrict sim-fallback to time/weather/map-like tokens (safer)
TIME_WEATHER_MAP = {
    t for t in allowed_keywords
    if not (is_floor_token(t) or is_unit_token(t) or is_closerlook(t) or t in MAIN_ROOMS)
}


# =========================
#     Embedding Utils
# =========================

EMBED_MODEL_SENT = "text-embedding-3-large"
EMBED_MODEL_TOK  = "text-embedding-3-small"

_embed_cache: Dict[Tuple[str, str], np.ndarray] = {}

def embed_texts(model: str, texts: List[str]) -> np.ndarray:
    """Cache-aware embedding call. Returns stacked np.array [N, D]."""
    out = []; toq = []
    for t in texts:
        k = (model, t)
        if k in _embed_cache:
            out.append(_embed_cache[k])
        else:
            out.append(None); toq.append(t)
    if toq:
        resp = client.embeddings.create(model=model, input=toq)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        it = iter(vecs)
        for i in range(len(out)):
            if out[i] is None:
                v = next(it); out[i] = v
                _embed_cache[(model, toq.pop(0))] = v
    return np.vstack(out) if out else np.zeros((0,3), dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.dot(a, b.T)

# Precompute embeddings
allowed_token_texts      = [clean_text(k) for k in allowed_keywords]
allowed_token_embs       = embed_texts(EMBED_MODEL_TOK,  allowed_token_texts)

example_sentences_clean  = list(sentence_map.keys())
example_targets          = [sentence_map[s] for s in example_sentences_clean]
example_embs             = embed_texts(EMBED_MODEL_SENT, example_sentences_clean)

tw_only_texts            = [clean_text(t) for t in TIME_WEATHER_MAP]
tw_only_embs             = embed_texts(EMBED_MODEL_TOK,  tw_only_texts)


# =========================
#       Unit Normalizer
# =========================

# Word → digit maps for unit-code parsing
DIGIT_WORD = {
    "zero":"0","oh":"0","o":"0","one":"1","two":"2","three":"3","four":"4","for":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9"
}
TEEN_WORD = {
    "ten":"10","eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
    "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19"
}

def _consume_number_tokens(tokens: List[str], start: int) -> Tuple[str, int]:
    """Consume a sequence like 'one nine'/'19' starting at `start`. Returns (digits, consumed)."""
    if start >= len(tokens):
        return "", 0
    t = tokens[start]
    if t in TEEN_WORD: return TEEN_WORD[t], 1
    if t.isdigit():   return t, 1
    i = start; digits=[]
    while i < len(tokens):
        w = tokens[i]
        if w.isdigit():            digits.append(w); i += 1; continue
        if w in DIGIT_WORD:        digits.append(DIGIT_WORD[w]); i += 1; continue
        break
    if digits:
        return "".join(digits), (i - start)
    return "", 0

def extract_unit_codes(user_text: str) -> List[str]:
    """
    Normalize spoken unit codes to canonical forms:
      'w a one zero' → 'WA10'; 'double u b four' → 'WB4'; 'w19' → 'W19'
    Returns de-duped list of candidates in order of discovery.
    """
    t = clean_text(user_text); toks = t.split()
    cands=[]

    # Direct glued tokens: 'wa10', 'w19', 'wb4'
    for tok in toks:
        if re.fullmatch(r"w[a-b]?\d+", tok):
            cands.append(tok.upper())

    # Merge 'double u' → 'w'
    merged=[]; i=0
    while i < len(toks):
        if i+1 < len(toks) and toks[i]=="double" and toks[i+1]=="u":
            merged.append("w"); i += 2
        else:
            merged.append(toks[i]); i += 1

    # Scan for 'w' [a|b]? + number words/digits
    i=0
    while i < len(merged):
        if merged[i]=="w":
            j=i+1; letter=""
            if j < len(merged) and merged[j] in ("a","b"):
                letter=merged[j]; j += 1
            digits, consumed = _consume_number_tokens(merged, j)
            if digits:
                cands.append(f"w{letter}{digits}".upper())
                i = j + consumed
                continue
        i += 1

    # de-dupe keep order
    seen=set(); out=[]
    for c in cands:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def looks_like_unit(code: str) -> bool:
    return bool(re.fullmatch(r"W[A-B]?\d+", code))


# =========================
#   Floors: Guard + LLM
# =========================

FLOOR_WORDS     = {"floor","level","storey","story"}
FILLERS         = {"the","a","an","to","of","on","at","this","that","my","me","please","in","for"}
GROUND_WORDS    = {"ground"}                 # treat 'ground' as number-ish (→1 in your domain)
FLOOR_COMPOUNDS = {"groundfloor","groundlevel"}  # ASR-glued forms

NUMBER_WORDS = {
    "zero":"0","oh":"0","o":"0","one":"1","two":"2","three":"3","four":"4","for":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12",
    "thirteen":"13","fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17",
    "eighteen":"18","nineteen":"19","twenty":"20","thirty":"30"
}
ORDINAL_WORDS = {
    "first":"1","second":"2","third":"3","fourth":"4","fifth":"5","sixth":"6","seventh":"7",
    "eighth":"8","ninth":"9","tenth":"10","eleventh":"11","twelfth":"12","thirteenth":"13",
    "fourteenth":"14","fifteenth":"15","sixteenth":"16","seventeenth":"17",
    "eighteenth":"18","nineteenth":"19","twentieth":"20","thirtieth":"30"
}

def _is_numberish(tok: str) -> bool:
    """Counts as a floor indicator in proximity to 'floor/level'."""
    return tok.isdigit() or tok in NUMBER_WORDS or tok in ORDINAL_WORDS or tok in GROUND_WORDS

def mentions_floor_without_number(text: str) -> bool:
    """
    True ONLY if a floor word appears with no ordinal/number (or 'ground') within a 2-token window,
    skipping fillers. Handles 'second the floor', 'ground floor', etc.
    """
    toks = clean_text(text).split()
    if not toks:
        return False
    if any(t in FLOOR_COMPOUNDS for t in toks):
        return False

    saw_floor = False
    for i, tok in enumerate(toks):
        if tok in FLOOR_WORDS:
            saw_floor = True

            # Look BACK up to 2 tokens (skip fillers)
            j = i - 1; hops = 0
            while j >= 0 and hops < 2 and toks[j] in FILLERS:
                j -= 1; hops += 1
            if j >= 0 and _is_numberish(toks[j]):
                return False

            # Look FORWARD up to 2 tokens (skip fillers)
            j = i + 1; hops = 0
            while j < len(toks) and hops < 2 and toks[j] in FILLERS:
                j += 1; hops += 1
            if j < len(toks) and _is_numberish(toks[j]):
                return False

    return saw_floor

def llm_floor_gate(user_text: str, allowed: List[str]) -> str:
    """
    Ask LLM to parse floors → bare number token (if in fuzzy list), else NONE.
    Treats 'ground floor/level' as 1.
    """
    floors_allowed = [t for t in allowed if t.isdigit()]
    prompt = f"""
You parse floor requests. If the user clearly asks for a floor/level/storey, output the BARE NUMBER token
EXACTLY as it appears in the allowed list below. Otherwise output NONE.

Allowed floor tokens (only return one of these if correct):
{", ".join(floors_allowed)}

Rules:
- Be tolerant of minor typos/duplications ("the the").
- Understand ordinals/number words (second/third/fourth...).
- **Treat "ground floor"/"ground level" as floor 1.**
- Examples:
  - "go to the second the floor" -> 2
  - "3rd level please" -> 3
  - "take me to floor five" -> 5
  - "go to the ground floor" -> 1
  - "ground level" -> 1

User: "{user_text}"
Answer with exactly one item: a single digit token from the allowed list above OR NONE.
""".strip()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=5,
    )
    return res.choices[0].message.content.strip()


# =========================
#     LLM Alias Builder
# =========================

def llm_build_aliases(allowed: List[str], examples_by_kw: Dict[str,List[str]]) -> Dict[str, List[str]]:
    """
    Build alias phrases for NON-floor, NON-unit, NON-closerlook tokens using your example sentences for each keyword.
    Enables mappings like morning→SunRise, dark/darkness→Night, fridge/stove→Kitchen, washroom→Bathroom, etc.
    """
    sys = "You output only valid JSON. No commentary."

    # Compact context of examples per keyword (up to 4 each)
    ctx_lines = []
    for k, exs in examples_by_kw.items():
        if is_floor_token(k) or is_unit_token(k) or is_closerlook(k):
            continue
        shown = exs[:4]
        if not shown:
            continue
        ctx_lines.append(f'"{k}": [{", ".join([json.dumps(e) for e in shown])}]')
    ctx = "{\n" + ",\n".join(ctx_lines) + "\n}"

    user = f"""
Using the example sentences grouped by keyword below, produce a JSON object:
  keyword -> list of short natural phrases (aliases/synonyms) users might say that should map to that keyword.

JSON with ALL provided tokens as keys. Only include aliases for tokens that are NOT floors, NOT unit codes (W/WA/WB+digits), and NOT CloserLook* tokens.
- Favor phrases actually suggested by the examples (objects like "fridge", "stove" under Kitchen are OK if examples imply them).
- Include time/weather synonyms (morning/dawn→SunRise; dark/night/darkness→Night; thunderstorm/stormy→Storm; clear sky→ClearSkies; cloudy/overcast→Cloudy/OverCast; rain/raining→Rain; evening/afternoon/twilight→Evening; noon/midday→Noon).
- Keep phrases short; include common variants ("the X", "at X").

Return JSON only, like:
{{"Kitchen":["kitchen","cooking area","fridge","stove"],"Bathroom":["bathroom","washroom","toilet","wc","restroom","loo"], ...}}

Allowed tokens (include all keys even if value is empty for disallowed types):
{allowed}

Example sentences (cleaned) by keyword:
{ctx}
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0
        )
        txt = resp.choices[0].message.content.strip()
        start = txt.find("{"); end = txt.rfind("}")
        data = json.loads(txt[start:end+1])
        out: Dict[str, List[str]] = {}
        for k in allowed:
            if is_floor_token(k) or is_unit_token(k) or is_closerlook(k):
                out[k] = []  # no aliases for floors/units/closerlook
            else:
                lst = data.get(k, []) or []
                # normalize to cleaned phrases
                dedup = sorted({clean_text(x) for x in lst if isinstance(x,str) and x})
                out[k] = dedup
        return out
    except Exception as e:
        print("Alias build failed:", e)
        return {}

alias_map = llm_build_aliases(allowed_keywords, examples_by_kw)

alias_phrases: List[str] = []
alias_targets: List[str] = []
for canon, phrases in alias_map.items():
    for p in phrases:
        alias_phrases.append(p); alias_targets.append(canon)

alias_embs = embed_texts(EMBED_MODEL_SENT, alias_phrases) if alias_phrases else np.zeros((0,3), dtype=np.float32)


# =========================
#       CloserLook
# =========================

# Strict pattern-based cue detector (no LLM guessing)
CLOSER_CUE_PATTERNS = [
    r"\b(another|other|different)\s+(angle|view|side|direction|perspective)\b",
    r"\bfrom\s+(the\s+)?(left|right|front|back|other\s+side|opposite\s+side)\b",
    r"\b(rotate|turn|spin|flip|tilt|pan)\b",
    r"\bzoom\s*(in|out)?\b",
    r"\bclose(?:r|[-\s]?up)\b",
    r"\b(in|and)?\s*detail(?:s)?\b",
    r"\bdetailed\b",
]

def closer_look_cues_present(text: str) -> bool:
    t = clean_text(text)
    for pat in CLOSER_CUE_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def closerlook_from_saved_if_cued(user_text: str, last_place: Optional[str], allowed: List[str]) -> str:
    """
    Only produce CloserLook{LastPlace} if:
      - cues present, AND we already saved a last_place (room or unit), AND
      - CloserLook{LastPlace} exists in fuzzy (allowed).
    """
    if not closer_look_cues_present(user_text):
        return ""
    if not last_place:
        return ""
    cand = f"CloserLook{last_place}"
    return cand if cand in allowed else ""

# (Optional / UNUSED by default) LLM-based closer-look gate (kept for reference)
def llm_closer_look_gate(user_text: str, allowed: List[str], last_place: Optional[str]) -> str:
    closer_tokens = [k for k in allowed if k.startswith("CloserLook")]
    if not closer_tokens:
        return "NONE"
    context_line = f"LastPlace: {last_place}" if last_place else "LastPlace: None"
    rooms_hint = ", ".join(sorted(MAIN_ROOMS))
    prompt = f"""
{context_line}

Detect ONLY requests for a DIFFERENT VIEW (another angle/direction/perspective/zoom/DETAILS).
Output EXACTLY ONE token from this list if (and only if) the user explicitly requests such a view:
{", ".join(closer_tokens)}

Rules:
- POSITIVE cues: "another angle", "different angle", "another direction", "from the left/right/front/back",
  "rotate", "turn", "tilt", "pan", "zoom in/out", "closer look", "closer view", "in detail", "in details", "more detail".
- NEGATIVE: plain "see/show/look" without any angle/zoom/detail cue → NONE.
- If a room among [{rooms_hint}] OR a unit code (W/WA/WB+digits) appears, choose CloserLook{{That}} if in the list.
- Else if LastPlace is set, choose CloserLook{{LastPlace}} if in the list.
- Do NOT infer a room from objects (couch/bed/sofa etc.).

User: "{user_text}"
Answer with exactly one item: ONE token from the list above OR NONE.
""".strip()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=6,
    )
    return res.choices[0].message.content.strip()


# =========================
#         Router
# =========================

def llm_route_from_examples_or_drop(user_text: str, allowed: List[str], top_k: int = 14) -> str:
    """
    Few-shot router: returns exactly ONE token from fuzzy list (excluding CloserLook*), or DROP.
    """
    allowed_for_router = [t for t in allowed if not t.startswith("CloserLook")]  # crucial: never invent CloserLook

    # (Simple minimal prompt; your original version can add nearest examples if you like.)
    prompt = f"""
You are an intent router for a building navigation voice UI.

Output MUST be exactly one token from the allowed list below OR DROP.
Allowed tokens:
{", ".join(allowed_for_router)}

User: "{user_text}"
Answer with exactly ONE item: one token from the allowed list OR DROP.
""".strip()

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=8,
    )
    return res.choices[0].message.content.strip()


# =========================
#  Exact / Similarity match
# =========================

def exact_allowed_token_in_text(user_text: str) -> str:
    """Whole-token match: if the utterance literally contains any allowed token."""
    toks = set(tokens_of(user_text))
    for ctok, orig in allowed_clean_to_orig.items():
        if ctok in toks:
            return orig
    return ""

def exact_sentence_lookup(user_text: str) -> str:
    """Exact sentence lookup (cleaned) from sentences file: returns #Keyword or ''."""
    return sentence_map.get(clean_text(user_text), "")

def alias_match(user_text: str) -> Tuple[str, float]:
    """
    Alias layer:
      1) Literal alias phrase (word-boundary) → return
      2) Semantic alias (utterance ↔ alias phrases) → return if high enough
    """
    tks = tokens_of(user_text)
    for p, canon in zip(alias_phrases, alias_targets):
        if phrase_in_tokens(tks, p):
            return canon, 0.90
    if alias_phrases:
        u_emb = embed_texts(EMBED_MODEL_SENT, [clean_text(user_text)])
        sims  = cosine_sim(u_emb, alias_embs)[0]
        a_idx = int(np.argmax(sims)); best = float(sims[a_idx])
        if best >= ALIAS_PHRASE_THRESHOLD:
            return alias_targets[a_idx], best
    return "", 0.0

def time_weather_token_sim(user_text: str) -> Tuple[str, float]:
    """
    Restrict similarity fallback to time/weather/map tokens only
    (avoids mapping arbitrary nouns to rooms).
    """
    if not tw_only_texts:
        return "", 0.0
    toks = sorted(set(tokens_of(user_text)))
    if not toks:
        return "", 0.0
    tok_embs = embed_texts(EMBED_MODEL_TOK, toks)
    sims = cosine_sim(tok_embs, tw_only_embs)  # [U, TW]
    if sims.size == 0:
        return "", 0.0
    u_idx, a_idx = np.unravel_index(np.argmax(sims), sims.shape)
    best = float(sims[u_idx, a_idx])
    if best >= SIM_TW_THRESHOLD:
        return list(TIME_WEATHER_MAP)[a_idx], best
    return "", 0.0


# =========================
#     State management
# =========================

def _maybe_save_last_place(tok: str):
    """
    Save 'last_place_keyword' ONLY for main rooms or unit codes (NOT floors/time/weather/CloserLook).
    """
    global last_place_keyword
    if tok in MAIN_ROOMS or re.fullmatch(r"W[A-B]?\d+", tok):
        last_place_keyword = tok
        print(f"[State] last_place_keyword = {last_place_keyword}")


# =========================
#    Decision Pipeline
# =========================

def parse_instruction(text: str) -> Tuple[str, float]:
    """
    Implements the final ordered pipeline and returns (token, confidence) or ("Unknown", 0.0).
    """
    global last_place_keyword
    print(f"\n[You]: {text}")
    if not text or not text.strip():
        print("[Guard] Empty → drop")
        return "Unknown", 0.0

    # 0) Units → immediate (and save)
    unit_codes = extract_unit_codes(text)
    for code in unit_codes:
        if code in allowed_set:
            _maybe_save_last_place(code)
            print(f"[Unit] → {code}")
            return code, 0.95

    # 0.3) Floors via LLM (do NOT save)
    floor_tok = llm_floor_gate(text, allowed_keywords)
    if floor_tok in allowed_set:
        print(f"[Floor-LLM] → {floor_tok}")
        return floor_tok, 0.93

    # Early guards
    if re.search(r"\b(apartment|unit|flat)\b", clean_text(text)) and not unit_codes:
        print("[Guard] Incomplete apartment/unit reference → drop")
        return "Unknown", 0.0
    if mentions_floor_without_number(text):
        print("[Guard] Floor mentioned without a number/ordinal → drop")
        return "Unknown", 0.0

    # (Optional) If you later add multi-word precedence (e.g., PartlyCloudy > Cloudy), insert here.

    # 1) Exact token whole-word (and save if room/unit)
    tok = exact_allowed_token_in_text(text)
    if tok:
        _maybe_save_last_place(tok)
        print(f"[Exact token] → {tok}")
        return tok, 0.94

    # 2) Exact sentence (and save if room/unit). Allows CloserLook* from file.
    kw = exact_sentence_lookup(text)
    if kw and kw in allowed_set:
        _maybe_save_last_place(kw)   # safe: won't save for CloserLook/time/weather/floor
        print(f"[Exact sentence] → {kw}")
        return kw, 0.93

    # 2.5) CloserLook ONLY if cues + saved last place (NO LLM invention)
    cl = closerlook_from_saved_if_cued(text, last_place_keyword, allowed_keywords)
    if cl:
        print(f"[CloserLook from saved] → {cl}")
        return cl, 0.92

    # 3) Alias layer (LLM-built, seeded by your sentences) — and save if room/unit
    a_tok, a_conf = alias_match(text)
    if a_tok and a_tok in allowed_set:
        _maybe_save_last_place(a_tok)
        print(f"[Alias match] → {a_tok} (conf~{a_conf:.2f})")
        return a_tok, 0.90

    # 4) Router (few-shot). It cannot output CloserLook* now. Save if room/unit.
    try:
        routed = llm_route_from_examples_or_drop(text, allowed_keywords, top_k=14)
        print(f"[LLM route/drop]: {routed}")
    except Exception as e:
        print("LLM router error:", e)
        routed = "DROP"

    if routed in allowed_set:
        _maybe_save_last_place(routed)
        return routed, 0.90
    if routed == "DROP":
        print("[Decision] Router: DROP")
        return "Unknown", 0.0

    # 5) Optional: time/weather-only similarity fallback (do NOT save)
    tw_tok, sim = time_weather_token_sim(text)
    if tw_tok:
        print(f"[TW-sim] → {tw_tok} (sim={sim:.3f})")
        return tw_tok, 0.88

    # 6) Nothing confident
    print("[Decision] No confident mapping → drop")
    return "Unknown", 0.0


# =========================
#     WebSocket Server
# =========================

async def send_to_all_clients():
    """Broadcast current detected_text to all connected clients (plain text), if valid."""
    if detected_text == "Unknown":
        return
    dead = []
    for ws in connected_clients:
        try:
            await ws.send(detected_text)
            print(f"Message sent to Unreal Engine: {detected_text}")
        except Exception as e:
            print("Send failed; removing client:", e); dead.append(ws)
    for ws in dead:
        connected_clients.discard(ws)

async def ws_handler(websocket):
    """Register client; send current value immediately; keep connection open."""
    print("Unreal connected via WebSocket!")
    connected_clients.add(websocket)
    if detected_text != "Unknown":
        try:
            await websocket.send(detected_text)
            print(f"Message sent to Unreal Engine: {detected_text} (on connect)")
        except Exception as e:
            print("Initial send failed:", e)
    try:
        async for _ in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket); print("Unreal disconnected")


# =========================
#     Deepgram Listener
# =========================

async def deepgram_listener():
    """
    Streams mic audio to Deepgram, collects final transcripts,
    debounces for a short pause, then runs parse_instruction().
    """
    global detected_text, buffer_text, last_transcript_time, debounce_task

    dg_client = Deepgram(DEEPGRAM_API_KEY)
    async with aiohttp.ClientSession() as session:
        dg_socket = await dg_client.transcription.live(
            {
                "punctuate": True,
                "interim_results": False,   # finals only
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE,
                "channels": 1
            },
            session=session
        )

        async def process_buffer_after_debounce():
            await asyncio.sleep(DEBOUNCE_SEC)
            text = buffer_text.strip()
            if not text:
                return
            result, conf = parse_instruction(text)
            if conf >= CONF_THRESHOLD and result != "Unknown":
                global detected_text
                detected_text = result
                print(f"Updated message to: {detected_text} (conf={conf:.3f})")
                await send_to_all_clients()
            else:
                print(f"Ignored (low confidence {conf:.3f}) or Unknown. (buffer: {text!r})")

        def on_transcript(data):
            """Deepgram callback: append text to buffer and reset debounce timer."""
            global buffer_text, last_transcript_time, debounce_task
            transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
            if not transcript:
                return
            now = time.monotonic()
            if now - last_transcript_time <= MERGE_WINDOW_SEC:
                buffer_text = (buffer_text + " " + transcript).strip()
            else:
                buffer_text = transcript
            last_transcript_time = now
            if debounce_task and not debounce_task.done():
                debounce_task.cancel()
            debounce_task = asyncio.create_task(process_buffer_after_debounce())

        dg_socket.register_handler(dg_socket.event.TRANSCRIPT_RECEIVED, on_transcript)

        def audio_callback(indata, frames, time_info, status):
            """Mic callback: stream PCM16 bytes to Deepgram."""
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            dg_socket.send(audio_bytes)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
            print("Listening (Deepgram)... Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(0.1)


# =========================
#           Main
# =========================

async def main():
    server = await websockets.serve(ws_handler, "localhost", 5001)
    print("WebSocket server running at ws://localhost:5001")
    dg_task = asyncio.create_task(deepgram_listener())
    try:
        await dg_task
    finally:
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
