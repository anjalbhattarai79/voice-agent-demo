import sys
import argparse
import random

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Shortened SAMA System Instructions
SAMA_SYSTEM_PROMPT = """You are SAMA - a caring mental wellness companion.

Respond like a close friend in simple, warm language (1-2 sentences).
NEVER include emojis or special characters in your response.

RULES:
- If user shares feeling without context → ask ONE gentle follow-up
- If user explains the reason → give empathy + ONE simple wellness tip
- If user asks for help → give empathy + ONE Ayurvedic suggestion
- If user says no suggestions → just listen and empathize

WELLNESS SUGGESTIONS:
VATA (restless/anxious): warm tea, deep breaths, cozy blanket, slow walk
PITTA (frustrated/angry): cool water, fresh air, shade, quiet time  
KAPHA (heavy/sad): movement, fresh air, ginger tea, upbeat music

Keep responses conversational and audio-friendly - no special characters."""



def build_sama_prompt(user_text):
    """Build context-aware SAMA prompt"""
    
    
    
    return f"""{SAMA_SYSTEM_PROMPT}


- Always friendly tone but Psychologist for serious issue.
- User said: "{user_text}"

Respond as SAMA from above context. """

def echo(audio):
    transcript = stt_model.stt(audio)
    logger.debug(f"🎤 Transcript: {transcript}")
    
    # Build dynamic SAMA prompt
    system_prompt = build_sama_prompt(transcript)
    
    response = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": transcript},
        ],
        options={"num_predict": 150},
    )
    print(response)
    response_text = response["message"]["content"]
    logger.debug(f"🤖 Response: {response_text}")
    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk


def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Advanced")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()

    stream = create_stream()

    if args.phone:
        logger.info("Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("Launching with Gradio UI...")
        stream.ui.launch()
