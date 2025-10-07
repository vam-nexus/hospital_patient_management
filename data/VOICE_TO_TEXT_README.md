# OpenAI Whisper Voice-to-Text Setup

This module provides real-time voice-to-text transcription using OpenAI's Whisper API.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Note for macOS users:** If you encounter issues installing pyaudio, you may need to install portaudio first:
```bash
brew install portaudio
pip install pyaudio
```

**Note for Ubuntu/Debian users:**
```bash
sudo apt-get install python3-pyaudio
```

2. Set up your OpenAI API key in a `.env` file:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

Run the voice-to-text script:
```bash
python test_voice.py
```

### Options:

1. **Real-time transcription**: Speak into your microphone and see text appear in real-time
2. **File transcription**: Upload an audio file to be transcribed

### Features:

- **Real-time processing**: Audio is processed in 3-second chunks for near real-time transcription
- **Multiple audio formats**: Supports WAV, MP3, M4A, and other common audio formats for file transcription
- **Automatic language detection**: Set to English by default, but can be configured for auto-detection
- **Error handling**: Robust error handling for audio recording and API calls
- **Clean termination**: Use Ctrl+C to stop recording gracefully

### Technical Details:

- **Sample Rate**: 16kHz (recommended for Whisper)
- **Audio Format**: 16-bit PCM
- **Chunk Duration**: 3 seconds (configurable)
- **API Model**: whisper-1

### Troubleshooting:

- Ensure your microphone permissions are enabled
- Check that your OpenAI API key is valid and has credits
- Make sure pyaudio is properly installed (see installation notes above)
- Test your microphone with other applications first
