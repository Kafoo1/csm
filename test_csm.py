# working_audio.py
import numpy as np
import wave

def create_speech_audio(text, filename="output.wav"):
    """Create working speech-like audio"""
    
    # Parameters
    sample_rate = 24000
    words = len(text.split())
    duration = max(words * 0.3, 1.0)  # At least 1 second
    samples = int(duration * sample_rate)
    
    # Generate audio
    t = np.linspace(0, duration, samples)
    
    # Multiple frequencies for speech
    audio = np.zeros(samples)
    for freq in [200, 400, 800]:
        audio += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Add envelope
    envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
    audio = audio * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.5
    
    # Convert to 16-bit
    audio_16bit = (audio * 32767).astype(np.int16)
    
    # Save
    with wave.open(filename, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_16bit.tobytes())
    
    print(f"✅ Created {filename} ({duration:.2f}s)")
    return filename

# Test
if __name__ == "__main__":
    texts = [
        "Hello, welcome to our service.",
        "I can help you book an appointment.",
        "Thank you for calling!"
    ]
    
    for i, text in enumerate(texts):
        create_speech_audio(text, f"working_{i}.wav")