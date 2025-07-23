# Hello, Gemma! 🗣️🤖

A fun, hackable voice assistant that lives on your machine. Say "ok google" or your own custom wake word and ask it anything! It listens, thinks with the power of Google's Gemma, and talks back to you. Built with Python and a sprinkle of AI magic. ✨

## Features

-   **Wake Word Detection:** Powered by Picovoice Porcupine. Use built-in wake words like "ok google" or train your own custom wake word (e.g., "Hey, my assistant") and use the generated `.ppn` file.
-   **Advanced Voice Activity Detection:** Uses WebRTC VAD to intelligently detect speech, recording your query and automatically stopping after a configurable duration of silence.
-   **AI-Powered Brain:** Uses Hugging Face Transformers to run the `google/gemma-3n-E2B-it` model for understanding and responding to your audio queries.
-   **Fast Text-to-Speech:** Responds with a synthesized voice using `pyttsx3` for speedy, local TTS.
-   **Verbal Cues:** Provides spoken feedback like "Listening" and "Processing" instead of just beeps.
-   **Customizable:** Easily tweak the wake word, AI model, VAD aggressiveness, and more!

## Requirements

-   Python 3.x
-   A Hugging Face account and access token.
-   A Picovoice account and access key.

## Installation

1.  **Clone this project** (or just save `hello_gemma.py`).

2.  **Install Python dependencies:**
    ```bash
    pip install pvporcupine sounddevice numpy torch transformers huggingface_hub pyttsx3 beepy webrtcvad
    ```

3.  **Install system libraries.** This might be needed for `sounddevice` and `pyttsx3`.
    
    For Debian/Ubuntu:
    ```bash
    sudo apt-get update
    sudo apt-get install portaudio19-dev libasound2-dev espeak
    ```

4.  **Get your access keys:**
    -   Sign up for a free [Picovoice Account](https://console.picovoice.ai/) to get your `pv_access_key`. You can also use the Picovoice Console to create your own custom wake word `.ppn` files.
    -   Sign up for a [Hugging Face Account](https://huggingface.co/join) to get your `hf_token`.

## Usage

Run the assistant from your terminal. You'll need to provide your access keys.

**Using a built-in wake word:**
```bash
python hello_gemma.py --hf_token YOUR_HUGGINGFACE_TOKEN --pv_access_key YOUR_PICOVOICE_ACCESS_KEY --wake_word "jarvis"
```

**Using a custom wake word file:**
```bash
python hello_gemma.py --hf_token YOUR_TOKEN --pv_access_key YOUR_KEY --keyword_path /path/to/your/custom_word.ppn
```

### Command-Line Arguments:

-   `--hf_token` (Required): Your Hugging Face access token.
-   `--pv_access_key` (Required): Your Picovoice access key.
-   `--wake_word`: A built-in wake word to listen for. Ignored if `--keyword_path` is provided. (Default: `ok google`)
-   `--keyword_path`: Path to a custom Porcupine keyword file (`.ppn`). Overrides `--wake_word`.
-   `--gemma_model_id`: The Gemma model ID to use. (Default: `google/gemma-3n-E2B-it`)
-   `--vad_aggressiveness`: WebRTC VAD aggressiveness (0-3). 0 is least aggressive, 3 is most. (Default: `1`)
-   `--silence_duration`: The duration of silence in seconds to stop recording. (Default: `5`)
-   `--device_name`: A substring of your microphone's name (e.g., "Jabra", "USB"). (Default: `Jabra`)

## How It Works

1.  The script starts and listens for the configured wake word.
2.  When you say it, it gives a verbal cue ("Listening.") and starts recording your voice.
3.  It keeps recording until it detects a configured duration of silence using WebRTC VAD.
4.  The recorded audio is sent to the Gemma model.
5.  Gemma processes the audio and generates a text response.
6.  The text response is spoken back to you using `pyttsx3`.
7.  The assistant goes back to listening for the wake word. Ready for your next command!
