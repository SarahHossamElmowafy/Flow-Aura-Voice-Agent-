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

# Deepgram & OpenAI API Keys
DEEPGRAM_API_KEY = ""
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

# Mic settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Shared state
detected_text = "Unknown"
last_main_keyword = None

# ------------------ Keywords ------------------

fuzzy_keywords = [
    "Living", "Kitchen", "Bathroom", "Master", "Guest",
    "CloserLookLiving", "CloserLookKitchen", "CloserLookBathroom", "CloserLookMaster", "CloserLookGuest",
    "Night", "Noon", "Live", "Evening", "SunRise",
    "ClearSkies", "Cloudy", "OverCast", "PartlyCloudy", "Rain", "Storm",
    "WA10", "W10", "WA3", "WB3", "WA1", "WA6", "WA2", "WA7", "W19", "WA9", "WB4", "WB1"
]

def load_sentence_keyword_map(path="sentence_to_keyword.txt"):
    mapping = {}
    current_keyword = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    current_keyword = line[1:].strip()
                    print(f"\n[Keyword section] {current_keyword}")
                elif line and current_keyword:
                    lower_sentence = line.lower()
                    mapping[lower_sentence] = current_keyword
                    print(f"  ↳ Example: '{line}'  -->  Keyword: '{current_keyword}'")
    except Exception as e:
        print("Error loading sentence_to_keyword.txt:", e)

    print("\n[Summary] Loaded", len(mapping), "sentences from", path)
    return mapping


sentence_map = load_sentence_keyword_map()

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def fuzzy_match_from_file(text, sentence_map, threshold=0.75):
    text = clean_text(text)
    for sentence, keyword in sentence_map.items():
        if sentence in text:
            print(f"[Substring match]: {sentence} -> {keyword}")
            return keyword
    matches = get_close_matches(text, sentence_map.keys(), n=1, cutoff=threshold)
    if matches:
        print(f"[Fuzzy match]: {matches[0]} -> {sentence_map[matches[0]]}")
        return sentence_map[matches[0]]
    return None

def is_vague(text):
    text = text.lower()
    return any(p in text for p in [
        "closer look", "show in detail", "zoom", "see better", "show it better",
        "show me more", "can i see it", "from another", "more details", "look closely"
    ])

def parse_instruction(text):
    global last_main_keyword
    print(f"\n[You]: {text}")
    cleaned = clean_text(text)

    match = fuzzy_match_from_file(cleaned, sentence_map)
    MAIN_ROOMS = {"Kitchen", "Master", "Living", "Guest", "Bathroom"}

    if match:
        print(f"[File matched]: {match}")
        if match in MAIN_ROOMS:
            last_main_keyword = match
        return match


    if is_vague(text) and last_main_keyword:
        guess = f"CloserLook{last_main_keyword}"
        if guess in fuzzy_keywords:
            print(f"[Inferred from last keyword]: {guess}")
            return guess

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
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        keyword = res.choices[0].message.content.strip()
        print(f"[GPT matched]: {keyword}")
        if keyword in fuzzy_keywords or keyword.upper() in [k.upper() for k in fuzzy_keywords]:
            if keyword in MAIN_ROOMS:
                last_main_keyword = keyword
            return keyword

    except Exception as e:
        print("GPT Error:", e)

    return "Unknown"

# ------------------ Deepgram Microphone Handler ------------------

async def deepgram_listener():
    global detected_text

    dg_client = Deepgram(DEEPGRAM_API_KEY)

    async with aiohttp.ClientSession() as session:
        dg_socket = await dg_client.transcription.live(
            {
                "punctuate": True,
                "interim_results": False,
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE,
                "channels": 1
            },
            session=session
        )

        def on_transcript(data):
            transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
            if transcript:
                result = parse_instruction(transcript)
                if result != "Unknown":
                    detected_text = result
                    print(f"Updated message to: {detected_text}")
                else:
                    print("Unrecognized command.")

        dg_socket.register_handler(dg_socket.event.TRANSCRIPT_RECEIVED, on_transcript)

        def callback(indata, frames, time, status):
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            dg_socket.send(audio_bytes)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SIZE):
            print("Listening (Deepgram)... Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(0.1)

# ------------------ WebSocket ------------------

async def handler(websocket):
    global detected_text
    print("Unreal connected via WebSocket.")
    last_sent = None
    try:
        while True:
            if detected_text != last_sent:
                await websocket.send(detected_text)
                print(f"Sent to Unreal: {detected_text}")
                last_sent = detected_text
            await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        print("Unreal disconnected")

async def websocket_server():
    async with websockets.serve(handler, "localhost", 5001):
        print("WebSocket server running at ws://localhost:5001")
        await asyncio.Future()

def start_websocket():
    def run():
        asyncio.run(websocket_server())
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# ------------------ Main ------------------

if __name__ == "__main__":
    start_websocket()
    try:
        asyncio.run(deepgram_listener())
    except KeyboardInterrupt:
        print("Stopped.")
