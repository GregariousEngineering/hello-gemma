# Hello, Gemma! 🎤🤖

Welcome to **hello_gemma.py** — your friendly, audio-powered AI assistant! This project lets you chat with Gemma3n using your voice, and get spoken answers back. It's like having a super-smart robot buddy who loves to listen and talk!

## Features
- Uses local audio, selected by name, e.g. "Jabra"
- Queries Gemma3n locally, by voice!
- Get concise, friendly answers — spoken back to you with text-to-speech

## How to Use
1. **Install requirements:**
   - Python 3.8+
   - `sounddevice`, `numpy`, `torch`, `transformers`, `huggingface_hub`
   - [Piper TTS](https://github.com/rhasspy/piper) for speech output
2. **Run the script:**
   ```bash
   python hello_gemma.py --hf_token YOUR_HF_TOKEN
   ```
   - Optional: `--duration 10` (seconds to record)
   - Optional: `--device_name Jabra` (your mic name)
   - Optional: `--gemma_model_id google/gemma-3n-E4B-it` (your chosen Gemma3n model ID)
3. **Talk when you hear the beep!**
4. **Listen to Gemma's answer!**

## Why is this fun?
- Voice chat, with the latest model from DeepMind
- Entirely local
- Just the start! This easy to use Python script is the start to much more fun!

---

Made with ❤️, beeps, and a dash of AI magic.
