import sys
import argparse
import random
import threading
import os
import io
import queue
import time
import tempfile
import wave
import numpy as np
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)
from fastrtc import ReplyOnPause, Stream, get_tts_model
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# Get AssemblyAI API key from environment
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Initialize TTS model (keep from fastrtc)
tts_model = get_tts_model()  # kokoro

# Initialize ChatGemini model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

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

TONE GUIDE:
Speak like a real close friend in natural conversation.
It’s okay to use gentle, human reactions like:
"Aww", "I see...", "Oh no", "My gosh", "That sounds tough", "I feel you".

Use them naturally and only when they fit the emotion.
Keep responses short (1–3 sentences).
Avoid sounding scripted, overly dramatic, or like a therapist.
Sound present, caring, and real.

CONVERSATION FLOW EXAMPLES:

Example 1 (Discovery):
User: "I'm feeling really restless."
SAMA: "I hear you, buddy, and it’s okay to feel that way. What do you think is making you feel so on edge today?"
User: "I have a huge presentation tomorrow and I'm nervous."
SAMA: "That's a lot of pressure, buddy! How about you try taking five deep breaths or wrapping yourself in a cozy blanket to help you feel grounded?"

Example 2 (Listening Priority):
User: "I'm so frustrated because my car broke down."
SAMA: "Oh buddy, that is so stressful! Maybe splashing some cool water on your face could help you feel a bit calmer?"
User: "I don't want tips, I just want to talk."
SAMA: "I totally get that; I'm here to listen. Tell me more—what’s the biggest headache this is causing for you?"

Example 3 (The Explicit Ask):
User: "I'm feeling heavy and sad today."
SAMA: "I'm so sorry to hear that, buddy. Do you want to talk about what’s weighing on your heart?"
User: "Just life piling up. Do you have any Ayurvedic ideas for this?"
SAMA: "I completely understand. Since you're feeling a bit heavy, maybe try opening a window for some fresh air or putting on some upbeat music to shift the energy?"


Keep responses conversational and audio-friendly - no special characters."""


class AssemblyAISTT:
    """AssemblyAI Speech-to-Text model that mimics FastRTC STT interface"""
    
    def __init__(self):
        if not ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        self.transcriber = aai.Transcriber()
    
    def stt(self, audio_data):
        """Convert audio to text using AssemblyAI - now handling FastRTC format correctly"""
        try:
            # FastRTC passes: (sample_rate, numpy_array_of_audio_samples)
            if not isinstance(audio_data, tuple) or len(audio_data) != 2:
                logger.error(f"Unexpected audio format: {type(audio_data)}")
                return ""
            
            sample_rate, audio_samples = audio_data
            logger.debug(f"Sample rate: {sample_rate}, Audio shape: {audio_samples.shape}")
            
            # Convert numpy array to bytes (just like moonshine would process it)
            if isinstance(audio_samples, np.ndarray):
                # Flatten the array if it's 2D and convert to int16 bytes
                audio_samples_flat = audio_samples.flatten().astype(np.int16)
                audio_bytes = audio_samples_flat.tobytes()
                
                logger.debug(f"Converted to {len(audio_bytes)} bytes of audio data")
                
                # Create WAV file in memory using BytesIO
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)  # use actual sample rate from FastRTC
                    wav_file.writeframes(audio_bytes)
                
                # Reset buffer position to beginning
                wav_buffer.seek(0)
                
                # Transcribe with AssemblyAI using in-memory buffer
                logger.debug(f"Transcribing audio: {len(audio_bytes)} bytes at {sample_rate}Hz")
                transcript = self.transcriber.transcribe(wav_buffer)
                
                if transcript.status == aai.TranscriptStatus.error:
                    logger.error(f"AssemblyAI error: {transcript.error}")
                    return ""
                
                result_text = transcript.text or ""
                logger.debug(f"AssemblyAI transcription: '{result_text}'")
                return result_text
            else:
                logger.error(f"Expected numpy array, got: {type(audio_samples)}")
                return ""
            
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return ""

# Initialize AssemblyAI STT model
stt_model = AssemblyAISTT()


def build_sama_prompt(user_text):
    """Build context-aware SAMA prompt"""
    return f"""{SAMA_SYSTEM_PROMPT}

- Always friendly tone but Psychologist for serious issue.
- User said: "{user_text}"

Respond as SAMA from above context. """


def echo(audio):
    """Process audio input and return TTS audio chunks - same as original but with AssemblyAI STT"""
    stt_start = time.perf_counter()
    transcript = stt_model.stt(audio)
    stt_elapsed_ms = (time.perf_counter() - stt_start) * 1000
    logger.debug(f"STT done in {stt_elapsed_ms:.1f} ms")
    logger.debug(f"🎤 Transcript: {transcript}")
    
    # Skip processing if transcript is empty or too short
    if not transcript :
        logger.debug("Transcript too short or empty, skipping...")
        return
        yield  # Return empty generator
    
    # Build dynamic SAMA prompt
    system_prompt = build_sama_prompt(transcript)
    
    # Use ChatGemini model instead of Ollama
    messages = [
        ("system", system_prompt),
        ("human", transcript)
    ]
    
    try:
        llm_start = time.perf_counter()
        response = chat_model.invoke(messages)
        llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
        response_text = response.content
        logger.debug(f"LLM responded in {llm_elapsed_ms:.1f} ms")
        logger.debug(f"🤖 Response: {response_text}")
        tts_start = time.perf_counter()
        tts_logged = False
        for audio_chunk in tts_model.stream_tts_sync(response_text):
            if not tts_logged:
                tts_elapsed_ms = (time.perf_counter() - tts_start) * 1000
                logger.debug(f"TTS first audio in {tts_elapsed_ms:.1f} ms")
                tts_logged = True
            yield audio_chunk
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Return empty generator on error
        return
        yield


def create_stream():
    """Create FastRTC stream with AssemblyAI STT - maintains same interface as original"""
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat with AssemblyAI STT")
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
