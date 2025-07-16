# Inspired by https://github.com/asierarranz/Google_Gemma_DevDay/blob/main/Gemma2/assistant.py
# https://ai.google.dev/gemma/docs/capabilities/audio

import os, sounddevice as sd, numpy as np, tempfile, wave
from huggingface_hub import login
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import argparse

ASSISTANT_PROMPT = """
You're an audio AI assistant, trained to understand and respond to audio queries.
Process and understand the user's query from the audio input provided and provide your best possible response.
Answer questions clearly and concisely in a friendly, professional tone in the same language they were asked.
Keep replies short to speed up inference. If unsure, admit it and suggest looking into it further.
"""

# Current directory and path for beep sound files (used to indicate recording start and end)
current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "assets/bip.wav")
bip2_sound = os.path.join(current_dir, "assets/bip2.wav")

# Find the device for audio recording by matching part of the device name
def find_device(device_name_substring):
    for i, device in enumerate(sd.query_devices()):
        if device_name_substring.lower() in device['name'].lower():
            return i
    raise ValueError(f"Device with name '{device_name_substring}' not found")

# Play sound (beep) to signal recording start/stop
def play_sound(sound_file):
    os.system(f"aplay {sound_file}")

# Record audio using sounddevice, save it as a .wav file
def record_audio(filename, duration=10, fs=16000, device_name="Jabra"):
    sd.default.device = find_device(device_name)
    play_sound(bip_sound)  # Start beep
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to complete
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    play_sound(bip2_sound)  # End beep

# Convert text to speech using Piper TTS model
def text_to_speech(text):
    os.system(f'echo "{text}" | /home/gregarious/piper/build/piper --model /usr/local/share/piper/models/en-us-lessac-medium.onnx --output_file response.wav && aplay response.wav')

# Send a query to huggingface transformers model and get a response
def ask_gemma(audio_file):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": f"{audio_file}"},
                {"type": "text", "text": f"{ASSISTANT_PROMPT}"},
            ]
        }
    ]
    input_ids = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True, return_dict=True,
            return_tensors="pt",
    )
    input_ids = input_ids.to(model.device, dtype=model.dtype)
    outputs = model.generate(**input_ids, max_new_tokens=64)
    text = processor.batch_decode(
        outputs,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )
    return text[0]

def parse_args():
    parser = argparse.ArgumentParser(description="Hello, Gemma! Gemma3n Audio Assistant")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--gemma_model_id", type=str, default="google/gemma-3n-E4B-it", help="Gemma model ID")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds")
    parser.add_argument("--device_name", type=str, default="Jabra", help="Audio device name substring")
    return parser.parse_args()

def main():
    args = parse_args()

    login(args.hf_token)  # Login to Hugging Face with provided token to pull model
    processor = AutoProcessor.from_pretrained(args.gemma_model_id, device_map="auto")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.gemma_model_id, torch_dtype="auto", device_map="auto")

    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            record_audio(tmpfile.name, duration=args.duration, device_name=args.device_name)  # Record the audio input
            print(f"Recorded audio saved to {tmpfile.name}")
            response = ask_gemma(tmpfile.name) # Send the recorded audio file to Gemma3n directly!
            print(f"Agent response: {response}")
            if response:
                text_to_speech(response)  # Convert response to speech

# Entry point of the script
if __name__ == "__main__":
    main()