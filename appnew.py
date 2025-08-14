"""
Voice-controlled Unreal Engine bridge:
- Captures microphone audio
- Streams it to Deepgram for live speech-to-text
- Parses transcripts into a constrained set of scene keywords
- Sends the latest detected keyword over a local WebSocket to Unreal

Key ideas:
- Two concurrent components:
  1) Deepgram live transcription + parsing
  2) WebSocket server that pushes the current keyword to Unreal whenever it changes

Notes:
- Never hardcode API keys in source for production. Prefer environment variables or a secrets manager.
- The keyword resolution pipeline favors:
    a) Exact/substring matches from a curated sentence→keyword map file
    b) Fuzzy string matching for near-misses
    c) An LLM "last resort" that is constrained to return one allowed keyword
- If user says something vague (e.g., "show me more") after a room was recently detected,
  it infers a "CloserLook<Room>" variant when possible.
"""

import asyncio
import sounddevice as sd
import numpy as np
import websockets
import threading
from difflib import get_close_matches
import re
from openai import OpenAI
from deepgram import Deepgram
import aiohttp

# ------------------ Configuration ------------------

# ⚠️ SECURITY: Avoid committing API keys. Use environment variables or a secrets manager in production.
# The tuples below look like (label, key). The OpenAI SDK expects just the key string when you pass it.
DEEPGRAM_API_KEY = "DEEPGRAM_API_KEY", ""
OPENAI_API_KEY   = "OPENAI_API_KEY",   ""

# Initialize OpenAI client. We pass only the key part (tuple index 1).
client = OpenAI(api_key=OPENAI_API_KEY[1])

# Microphone capture settings
SAMPLE_RATE = 16000        # 16 kHz is standard for speech models
CHUNK_DURATION = 0.5       # seconds of audio per chunk callback
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # number of samples per chunk

# Shared state across coroutines/threads
detected_text = "Unknown"  # latest resolved keyword sent to Unreal
last_main_keyword = None   # remembers last main room (for "closer look" inference)

# ------------------ Keywords ------------------

# Allowed, canonical keywords for Unreal scenes. Keep this authoritative list in sync with Unreal.
fuzzy_keywords = [
    "Living", "Kitchen", "Bathroom", "Master", "Guest",
    "CloserLookLiving", "CloserLookKitchen", "CloserLookBathroom", "CloserLookMaster", "CloserLookGuest",
    "Night", "Noon", "Live", "Evening", "SunRise",
    "ClearSkies", "Cloudy", "OverCast", "PartlyCloudy", "Rain", "Storm",
    "WA10", "W10", "WA3", "WB3", "WA1", "WA6", "WA2", "WA7", "W19", "WA9", "WB4", "WB1"
]

def load_sentence_keyword_map(path: str = "sentence_to_keyword.txt") -> dict:
    """
    Load a text file that maps example user sentences to a target keyword.
    File format:
      #KeywordName
      Example sentence 1
      Example sentence 2
      ...
      #NextKeyword
      Another sentence
      ...

    Lines starting with '#' define the current keyword section.
    All subsequent non-empty lines are examples mapped to that keyword.
    """
    mapping = {}
    current_keyword = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    # Start a new keyword section
                    current_keyword = line[1:].strip()
                    print(f"\n[Keyword section] {current_keyword}")
                elif line and current_keyword:
                    # Map the lowercased example sentence to the current keyword
                    lower_sentence = line.lower()
                    mapping[lower_sentence] = current_keyword
                    print(f"  ↳ Example: '{line}'  -->  Keyword: '{current_keyword}'")
    except Exception as e:
        # Non-fatal: continue with empty mapping (LLM fallback/fuzzy matching will still work)
        print("Error loading sentence_to_keyword.txt:", e)

    print("\n[Summary] Loaded", len(mapping), "sentences from", path)
    return mapping

# Preload sentence→keyword examples (fast lookup during runtime)
sentence_map = load_sentence_keyword_map()

def clean_text(text: str) -> str:
    """Normalize text for matching: lowercase and strip punctuation/extra spaces."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def fuzzy_match_from_file(text: str, sentence_map: dict, threshold: float = 0.75) -> str | None:
    """
    Try to resolve a user utterance to a keyword using:
      1) Substring match against known example sentences (robust to extra words)
      2) Fuzzy closest match (difflib) if no substring match found

    Returns the matched keyword or None.
    """
    text = clean_text(text)

    # 1) Substring match: if any known example sentence appears within the utterance
    for sentence, keyword in sentence_map.items():
        if sentence in text:
            print(f"[Substring match]: {sentence} -> {keyword}")
            return keyword

    # 2) Fuzzy match against the example set
    matches = get_close_matches(text, sentence_map.keys(), n=1, cutoff=threshold)
    if matches:
        print(f"[Fuzzy match]: {matches[0]} -> {sentence_map[matches[0]]}")
        return sentence_map[matches[0]]

    return None

def is_vague(text: str) -> bool:
    """
    Heuristic to detect vague, follow-up requests that likely mean "show more detail"
    about the last referenced room/scene.
    """
    text = text.lower()
    return any(p in text for p in [
        "closer look", "show in detail", "zoom", "see better", "show it better",
        "show me more", "can i see it", "from another", "more details", "look closely"
    ])

def parse_instruction(text: str) -> str:
    """
    Core NLP routing:
    - Log user utterance
    - Try rule-based resolution (file-backed examples + fuzzy)
    - If utterance is vague and a main room was recently seen, infer CloserLook<Room>
    - Fallback to an LLM constrained to return one allowed keyword
    - Return "Unknown" if no resolution
    """
    global last_main_keyword
    print(f"\n[You]: {text}")
    cleaned = clean_text(text)

    # Attempt deterministic matching from examples/fuzzy
    match = fuzzy_match_from_file(cleaned, sentence_map)
    MAIN_ROOMS = {"Kitchen", "Master", "Living", "Guest", "Bathroom"}

    if match:
        print(f"[File matched]: {match}")
        # Update last main room if applicable (used for "CloserLook" inference)
        if match in MAIN_ROOMS:
            last_main_keyword = match
        return match

    # Handle vague follow-ups using the last known main room
    if is_vague(text) and last_main_keyword:
        guess = f"CloserLook{last_main_keyword}"
        if guess in fuzzy_keywords:
            print(f"[Inferred from last keyword]: {guess}")
            return guess

    # LLM fallback: strictly ask for ONE keyword from the allowed list
    prompt = f"""
    You are a keyword-matching assistant for a voice-controlled Unreal Engine app.

    Here is a list of valid scene labels:
    {', '.join(fuzzy_keywords)}

    Rules:
    - Only return one exact keyword from the list.
    - If the keyword starts with 'W', return it in ALL CAPS.
    - Otherwise, use CamelCase.
    - Do NOT say anything else.

    User said: "{text}"
    """.strip()

    try:
        # Use chat.completions to request a single keyword; model choice can be upgraded if needed
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        keyword = res.choices[0].message.content.strip()
        print(f"[GPT matched]: {keyword}")

        # Accept only if it's in the allowed set (case-insensitive guard)
        if keyword in fuzzy_keywords or keyword.upper() in [k.upper() for k in fuzzy_keywords]:
            if keyword in MAIN_ROOMS:
                last_main_keyword = keyword
            return keyword

    except Exception as e:
        # Network or API error should not crash the pipeline; fall back to "Unknown"
        print("GPT Error:", e)

    return "Unknown"

# ------------------ Deepgram Microphone Handler ------------------

async def deepgram_listener():
    """
    Opens a live transcription session with Deepgram and streams microphone audio.
    - Converts float32 audio frames to 16-bit PCM bytes for Deepgram.
    - Receives transcript events and resolves them to scene keywords.
    - Updates the global 'detected_text' whenever a valid keyword is found.
    """
    global detected_text

    # Initialize Deepgram SDK client with your API key string
    dg_client = Deepgram(DEEPGRAM_API_KEY[1])

    async with aiohttp.ClientSession() as session:
        # Open a live transcription socket with the desired audio and NLP options
        dg_socket = await dg_client.transcription.live(
            {
                "punctuate": True,          # adds punctuation to transcripts
                "interim_results": False,   # only final results (simplifies downstream parsing)
                "encoding": "linear16",     # PCM 16-bit
                "sample_rate": SAMPLE_RATE, # must match microphone capture rate
                "channels": 1               # mono
            },
            session=session
        )

        # Callback when Deepgram produces a transcript JSON payload
        def on_transcript(data):
            # Defensive JSON access; Deepgram returns channel → alternatives → transcript
            transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
            if transcript:
                # Resolve transcript into a scene keyword
                result = parse_instruction(transcript)
                if result != "Unknown":
                    detected_text = result
                    print(f"Updated message to: {detected_text}")
                else:
                    print("Unrecognized command.")

        # Register the transcript handler with the socket
        dg_socket.register_handler(dg_socket.event.TRANSCRIPT_RECEIVED, on_transcript)

        # Audio callback: called by sounddevice with each audio chunk
        def callback(indata, frames, time, status):
            # indata: float32 in [-1.0, 1.0]; convert to 16-bit PCM bytes
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            dg_socket.send(audio_bytes)

        # Open microphone input stream; this blocks until cancelled
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SIZE):
            print("Listening (Deepgram)... Press Ctrl+C to stop.")
            while True:
                # Yield to event loop; keeps this coroutine alive
                await asyncio.sleep(0.1)

# ------------------ WebSocket ------------------

async def handler(websocket):
    """
    WebSocket connection handler for Unreal:
    - Sends the latest 'detected_text' whenever it changes from the last sent value.
    - Runs in a loop until the Unreal client disconnects.
    """
    global detected_text
    print("Unreal connected via WebSocket.")
    last_sent = None
    try:
        while True:
            # Only push when there's a change to limit chatter
            if detected_text != last_sent:
                await websocket.send(detected_text)
                print(f"Sent to Unreal: {detected_text}")
                last_sent = detected_text
            await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        print("Unreal disconnected")

async def websocket_server():
    """
    Starts a WebSocket server on ws://localhost:5001 and runs indefinitely.
    Unreal should connect to this endpoint to receive keyword updates.
    """
    async with websockets.serve(handler, "localhost", 5001):
        print("WebSocket server running at ws://localhost:5001")
        # Keep the server task alive forever
        await asyncio.Future()

def start_websocket():
    """
    Launch the WebSocket server in a background daemon thread so that the
    main thread can run the Deepgram listener event loop.
    """
    def run():
        asyncio.run(websocket_server())
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# ------------------ Main ------------------

if __name__ == "__main__":
    # Start WebSocket server (background thread)
    start_websocket()
    try:
        # Run Deepgram listener in the main asyncio loop
        asyncio.run(deepgram_listener())
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("Stopped.")
