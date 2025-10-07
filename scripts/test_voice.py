import os
import io
import wave
import pyaudio
import threading
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RealTimeWhisper:
    def __init__(self, chunk_duration=3, sample_rate=16000, channels=1):
        """
        Initialize the real-time Whisper transcription system.

        Args:
            chunk_duration (int): Duration of each audio chunk in seconds
            sample_rate (int): Audio sample rate (16kHz recommended for Whisper)
            channels (int): Number of audio channels (1 for mono)
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1024  # Frames per buffer
        self.format = pyaudio.paInt16

        # Audio recording variables
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_buffer = []

        # Threading for continuous recording
        self.recording_thread = None
        self.transcription_thread = None

        print(f"Initialized RealTimeWhisper:")
        print(f"  - Chunk duration: {chunk_duration} seconds")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Channels: {channels}")

    def start_recording(self):
        """Start the real-time audio recording and transcription."""
        if self.is_recording:
            print("Recording is already in progress!")
            return

        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            self.is_recording = True
            print("🎤 Recording started... Speak into your microphone!")
            print("Press Ctrl+C to stop recording")

            # Start recording thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()

            # Start transcription thread
            self.transcription_thread = threading.Thread(
                target=self._process_transcription
            )
            self.transcription_thread.daemon = True
            self.transcription_thread.start()

        except Exception as e:
            print(f"Error starting recording: {e}")
            self.stop_recording()

    def stop_recording(self):
        """Stop the audio recording and clean up resources."""
        if not self.is_recording:
            return

        self.is_recording = False
        print("\n🛑 Stopping recording...")

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()
        print("Recording stopped and resources cleaned up.")

    def _record_audio(self):
        """Continuously record audio in chunks."""
        frames = []
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)

        while self.is_recording:
            try:
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

                # When we have enough frames for the chunk duration, process it
                if len(frames) * self.chunk_size >= frames_per_chunk:
                    # Convert frames to audio buffer
                    audio_data = b"".join(frames)
                    self.audio_buffer.append(audio_data)
                    frames = []  # Reset for next chunk

            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def _process_transcription(self):
        """Process audio chunks for transcription."""
        while self.is_recording:
            if self.audio_buffer:
                # Get the oldest audio chunk
                audio_data = self.audio_buffer.pop(0)

                # Transcribe the audio chunk
                transcription = self._transcribe_audio(audio_data)

                if transcription and transcription.strip():
                    print(f"📝 Transcribed: {transcription}")
            else:
                # Sleep briefly if no audio to process
                time.sleep(0.1)

    def _transcribe_audio(self, audio_data):
        """
        Transcribe audio data using OpenAI Whisper API.

        Args:
            audio_data (bytes): Raw audio data

        Returns:
            str: Transcribed text
        """
        try:
            # Create a WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)

            # Reset buffer position for reading
            wav_buffer.seek(0)

            # Send to OpenAI Whisper API
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", wav_buffer.read(), "audio/wav"),
                language="en",  # You can change this or set to None for auto-detection
            )

            return response.text

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

    def transcribe_file(self, audio_file_path):
        """
        Transcribe an existing audio file.

        Args:
            audio_file_path (str): Path to the audio file

        Returns:
            str: Transcribed text
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",  # You can change this or set to None for auto-detection
                )
                return response.text
        except Exception as e:
            print(f"Error transcribing file {audio_file_path}: {e}")
            return None


def main():
    """Main function to demonstrate the voice-to-text functionality."""
    print("=== OpenAI Whisper Real-Time Voice-to-Text ===")
    print("Choose an option:")
    print("1. Real-time voice transcription")
    print("2. Transcribe an audio file")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # Real-time transcription
        whisper = RealTimeWhisper(chunk_duration=3)  # 3-second chunks

        try:
            whisper.start_recording()

            # Keep the main thread alive
            while whisper.is_recording:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            whisper.stop_recording()

    elif choice == "2":
        # File transcription
        file_path = input("Enter the path to your audio file: ").strip()

        if os.path.exists(file_path):
            whisper = RealTimeWhisper()
            print(f"Transcribing file: {file_path}")
            transcription = whisper.transcribe_file(file_path)

            if transcription:
                print(f"\n📝 Transcription:\n{transcription}")
            else:
                print("Failed to transcribe the file.")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")


if __name__ == "__main__":
    main()
