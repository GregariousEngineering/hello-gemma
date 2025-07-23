# https://ai.google.dev/gemma/docs/capabilities/audio

import os, sounddevice as sd, numpy as np, tempfile, wave, struct, pvporcupine, pyttsx3, beepy as bp, webrtcvad
from huggingface_hub import login
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForSpeechSeq2Seq
import argparse

# Global variables for model and processor
processor = None
model = None
tts_engine = None
vad_engine = None

ASSISTANT_PROMPT = """
You're an audio AI assistant, trained to understand and respond to audio queries.
Process and understand the user's query from the audio input provided and provide your best possible response.
Answer questions clearly and concisely in a friendly, professional tone in the same language they were asked.
Keep replies short to speed up inference. If unsure, admit it and suggest looking into it further.
"""

# Find the device for audio recording by matching part of the device name
def find_device(device_name_substring):
    for i, device in enumerate(sd.query_devices()):
        if device_name_substring.lower() in device['name'].lower():
            return i
    raise ValueError(f"Device with name '{device_name_substring}' not found")

# Play sound (beep) to signal recording start/stop
def play_sound(sound):
    bp.beep(sound=sound)

# Record audio using sounddevice, save it as a .wav file
def record_audio(filename, stream, duration=5, fs=16000):
    text_to_speech(f"Recording for {duration} seconds.")
    print(f"Recording for {duration}s.")
    
    num_frames_to_read = int(duration * fs)
    
    recorded_frames = []
    frames_read = 0
    
    stream.start()
    while frames_read < num_frames_to_read:
        chunk, overflowed = stream.read(stream.blocksize)
        recorded_frames.append(chunk)
        frames_read += len(chunk)
    stream.stop()

    audio = np.concatenate(recorded_frames, axis=0)
    audio = audio[:num_frames_to_read] # Trim to exact duration

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    text_to_speech("Processing.")

# Record audio using sounddevice, save it as a .wav file
def record_audio_vad(filename, stream, fs=16000, silence_duration_s=5):
    text_to_speech("Listening.")
    
    # WebRTC VAD requires 16-bit PCM audio frames of 10, 20, or 30 ms.
    frame_duration_ms = 30
    chunk_size = int(fs * frame_duration_ms / 1000) # 480 samples for 16kHz
    
    silence_frame_limit = int(silence_duration_s * 1000 / frame_duration_ms)
    
    recorded_frames = []
    silent_frames_count = 0
    
    stream.start()
    print(f"Recording until {silence_duration_s}s of non-speech.")
    while True:
        audio_chunk, overflowed = stream.read(chunk_size)
        
        is_speech = vad_engine.is_speech(audio_chunk.tobytes(), fs)
        
        if not is_speech:
            silent_frames_count += 1
        else:
            silent_frames_count = 0

        recorded_frames.append(audio_chunk)
        
        if silent_frames_count > silence_frame_limit:
            break
    stream.stop()
    
    # Trim the trailing silence
    if len(recorded_frames) > silence_frame_limit:
        audio = np.concatenate(recorded_frames[:-silence_frame_limit], axis=0)
        recording_duration = (len(recorded_frames)-silence_frame_limit) * frame_duration_ms
    else:
        audio = np.concatenate(recorded_frames, axis=0)
        recording_duration = len(recorded_frames) * frame_duration_ms

    print(f"Recorded {recording_duration / 1000.0} seconds.")

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # int16 is 2 bytes
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    
    text_to_speech("Processing.")

# Convert text to speech using pyttsx3
def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

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
    parser.add_argument("--pv_access_key", type=str, required=True, help="Picovoice Porcupine access key")
    parser.add_argument("--wake_word", type=str, default="ok google", help="Built-in wake word. Ignored if --keyword_path is provided.")
    parser.add_argument("--keyword_path", type=str, help="Path to a custom Porcupine keyword file (.ppn).")
    parser.add_argument("--gemma_model_id", type=str, default="google/gemma-3n-E2B-it", help="Gemma model ID")
    parser.add_argument("--vad_aggressiveness", type=int, default=1, choices=range(4), help="WebRTC VAD aggressiveness (0-3)")
    parser.add_argument("--silence_duration", type=int, default=5, help="Silence duration in seconds to stop recording")
    parser.add_argument("--device_name", type=str, default="Jabra", help="Audio device name substring")
    return parser.parse_args()

def main():
    global processor, model, tts_engine, vad_engine
    args = parse_args()

    login(args.hf_token)  # Login to Hugging Face with provided token to pull model
    processor = AutoProcessor.from_pretrained(args.gemma_model_id, device_map="sequential")
    model = AutoModelForImageTextToText.from_pretrained(args.gemma_model_id, torch_dtype="auto", device_map="sequential")
    tts_engine = pyttsx3.init()
    vad_engine = webrtcvad.Vad(args.vad_aggressiveness)

    porcupine = None
    audio_stream = None

    try:
        wake_word_for_print = ""
        if args.keyword_path:
            if not os.path.exists(args.keyword_path):
                raise ValueError(f"Keyword file not found at {args.keyword_path}")
            porcupine = pvporcupine.create(
                access_key=args.pv_access_key,
                keyword_paths=[args.keyword_path]
            )
            wake_word_for_print = os.path.basename(args.keyword_path)
        else:
            porcupine = pvporcupine.create(
                access_key=args.pv_access_key,
                keywords=[args.wake_word]
            )
            wake_word_for_print = args.wake_word

        sd.default.device = find_device(args.device_name)
        
        audio_stream = sd.InputStream(
            samplerate=porcupine.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=porcupine.frame_length
        )
        audio_stream.start()
        print(f"Listening for '{wake_word_for_print}'...")

        while True:
            pcm, _ = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm.tobytes())
            
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wake word detected!")
                audio_stream.stop() # Stop listening for wake word

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    record_audio_vad(tmpfile.name, audio_stream, fs=porcupine.sample_rate, silence_duration_s=args.silence_duration)
                    #record_audio(tmpfile.name, audio_stream, fs=porcupine.sample_rate)
                    print(f"Recorded audio saved to {tmpfile.name}")
                    response = ask_gemma(tmpfile.name) # Send the recorded audio file to Gemma3n directly!
                    print(f"Agent response: {response}")
                    if response:
                        text_to_speech(response)  # Convert response to speech
                
                print(f"Listening for '{wake_word_for_print}'...")
                audio_stream.start() # Resume listening

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if audio_stream is not None:
            audio_stream.close()
        if porcupine is not None:
            porcupine.delete()

# Entry point of the script
if __name__ == "__main__":
    main()