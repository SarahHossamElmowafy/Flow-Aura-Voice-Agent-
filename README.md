# Voice → Keyword → Unreal Bridge

Deterministic-first, LLM-assisted voice intent router that turns raw mic audio into **exact, pre-approved tokens** and streams them to Unreal via WebSocket.

> **Goal:** hear anything the user says, emit **exactly one** token from `fuzzy_keywords.txt` (or nothing), and keep behavior predictable and low-latency.

---

## ✨ What it does

1. **Listen** to the microphone and stream PCM audio to **Deepgram** (final transcripts only).
2. After a short pause (debounce), run the transcript through an **ordered pipeline**:
   - **0) Units**: normalize “w a one zero” → `WA10` (if it’s in fuzzy, return immediately).
   - **0.3) Floors**: small LLM gate parses “second the floor”, “ground floor” → bare digit (`"2"`, `"1"`). Returned **only** if that digit exists in fuzzy.
   - **0.5) CloserLook (STRICT)**: if the utterance has explicit angle/zoom/detail cues **and** there’s a saved **last place** (room or unit), emit `CloserLook{LastPlace}`. Also accepts *exact sentences* under `#CloserLook...` in the examples file.
   - **1) Exact token**: if the text contains any token from fuzzy (whole-word), return it.
   - **2) Exact sentence**: if the text exactly matches a sentence in `sentence_to_keyword.txt`, return that sentence’s `#Keyword` **iff** it exists in fuzzy.
   - **2.5) Aliases**: short phrases learned from your examples (literal or semantic) map to allowed tokens.
   - **3) Router**: few-shot LLM picks **one** allowed token or `DROP` (never invents `CloserLook*`).
   - **4) Time/Weather-only similarity fallback** (safe, limited set).
3. **Guardrails**: drop incomplete “apartment”/“floor” mentions. Only emit tokens that exist in `fuzzy_keywords.txt`.
4. **Broadcast** the chosen token (plain text) to Unreal over `ws://localhost:5001` when it changes.

---

## 🧠 Token emission rules (the contract)

- **Only** tokens present in `fuzzy_keywords.txt` are ever emitted.
- `CloserLook***` is emitted **only** when:
  1) the utterance exactly matches a sentence under `#CloserLook...` in `sentence_to_keyword.txt`, **or**
  2) the utterance contains **explicit closer-look cues** (another angle/side, zoom, in detail(s), …) **and** a **saved last place** exists → emit `CloserLook{LastPlace}`.
- “Last place” is saved **only** when we emit a **room** (`Kitchen`, `Bathroom`, `Living`, `Master`, `Guest`) or a **unit** (`W/WA/WB+digits`). Not saved for floors/time/weather/CloserLook.
- **Units short-circuit**: once a unit is detected (e.g., `WA10`), return it immediately and save it.
- **Floors**: LLM converts ordinal/words (“second”, “ground floor”) to bare digits (`"2"`, `"1"`) only if present in fuzzy. Mentions of floor/level **without** a number/ordinal are dropped.

---

## 📁 Project files

```
voice4.py                  # Main application
fuzzy_keywords.txt         # Allowed tokens (the only outputs we ever emit)
sentence_to_keyword.txt    # Training examples grouped under #Keyword headings
```

### `fuzzy_keywords.txt` (examples)

```
# Rooms
Kitchen
Bathroom
Living
Master
Guest

# Units
W10
WA1
WA3
WA6
WA7
WA9
WA10
W19
WB1
WB3
WB4

# Floors
1
2
3
4
5

# Time/Weather/Map/etc
Map
SunRise
Night
Cloudy
PartlyCloudy
Rain
Storm
...
# CloserLook tokens
CloserLookKitchen
CloserLookBathroom
CloserLookLiving
CloserLookMaster
CloserLookGuest
CloserLookWA10
CloserLookWB4
...
```

### `sentence_to_keyword.txt` (examples)

```
# Kitchen
Take me to the kitchen
Where is the kitchen
Show me the kitchen
...

# Bathroom
Take me to the bathroom
Where is the bathroom
...

# CloserLookBathroom
Show me a closer look at the bathroom
Let me see the bathroom in detail
...

# SunRise
I want to see it in the morning
At sunrise
...
```

> **Tip:** Don’t invent new output tokens here. The `#Keyword` must exist in `fuzzy_keywords.txt` or it will be ignored.

---

## 🚀 Quickstart

1. **Python** 3.10+ (3.11 recommended).
2. Install deps:
   ```bash
   pip install openai deepgram-sdk sounddevice websockets aiohttp numpy
   ```
3. Export keys (avoid hardcoding!):
   ```bash
   # macOS/Linux
   export OPENAI_API_KEY=sk-...
   export DEEPGRAM_API_KEY=dg-...

   # Windows PowerShell
   setx OPENAI_API_KEY "sk-..."
   setx DEEPGRAM_API_KEY "dg-..."
   ```
4. Add/curate your `fuzzy_keywords.txt` and `sentence_to_keyword.txt`.
5. Run:
   ```bash
   python voice4.py
   ```
6. Connect Unreal to **`ws://localhost:5001`** and read the plain-text token.

> **Audio permissions:** On macOS you may need to grant terminal/mic access. On Windows, ensure an input device is active (WASAPI).

---

## ⚙️ Configuration knobs

- **Latency**:
  - `DEBOUNCE_SEC` (default `1.0`): lower for snappier responses (may risk false triggers).
  - `MERGE_WINDOW_SEC` (default `2.3`): how long to merge consecutive finals into one utterance.
  - `CHUNK_DURATION` (default `0.5`): affects send cadence to Deepgram.
  - Deepgram `interim_results=False` keeps it simple; flip to `True` + endpointing if you need ultra low latency.
- **Thresholds**:
  - `ALIAS_PHRASE_THRESHOLD` (default `0.80`): utterance ↔ alias phrase.
  - `SIM_TW_THRESHOLD` (default `0.76`): safety gate for time/weather/map similarity fallback.

---

## 🔍 Behavior examples

- “**take me to the kitchen**” → `Kitchen` *(saved)* → “**in details**” → `CloserLookKitchen`
- “**second the floor**” → `2`; “**floor**” (no number) → **DROP**
- “**apartment** …” (no unit code) → **DROP**
- “**w a one zero**” → `WA10` *(saved)*
- “**washroom**” → `Bathroom` (alias)
- “**I wanna see it when it’s stormy**” → `Storm` (alias)
- “**partly cloudy**” → `PartlyCloudy` (ensure both words exist in aliases/examples/fuzzy)

---

## 🧩 How it works (architecture)

1. **Deterministic surface** (units → floors → guards → exact token → exact sentence).  
2. **Contextual surface**: closer-look only if **cues+saved last place** or **exact closer-look sentence**.  
3. **Assist surface**: alias mapper (literal/semantic) and finally a small **router** LLM that outputs **one allowed token or DROP**.  
4. **Safety rails**: similarity fallback limited to time/weather/map tokens; everything else must be literal, example-backed, or routed.

The system never emits a token that is not in `fuzzy_keywords.txt`.

---

## 🧪 Extending

- **Add new tokens**: put them in `fuzzy_keywords.txt`.  
- **Teach the model**: add grouped sentences under `#YourToken` in `sentence_to_keyword.txt`.  
- **Prefer multi‑word specificity**: if you have both `Cloudy` and `PartlyCloudy`, provide multi-word examples/aliases so `PartlyCloudy` wins when both words are present.

---

## 🛡️ Security & privacy

- Prefer environment variables for API keys (don’t commit keys).
- Logs print tokens and short state only; avoid logging sensitive transcripts in production.
- Consider disabling verbose logs in release builds.

---

## 🔮 Next step (planned)

**Connect floors ↔ units** with an optional `unit_to_floor.json` mapping. Rules: unit wins if unit+floor appear together; inferred floor used internally (not emitted); log mismatches/unknowns. See Linear ticket for acceptance criteria.

---

## 📜 License

Internal project. Add a license if you plan to distribute.
