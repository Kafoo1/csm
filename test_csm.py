from generator import load_csm_1b
import torchaudio

# Load the model
model = load_csm_1b(device="cuda")  # or "cpu" if no GPU

# Generate speech
audio = model.generate(
    text="Hello, this is a test of the voice system.",
    speaker=0,
    context=[],
    max_audio_length_ms=5000
)

# Save it
torchaudio.save("test.wav", audio.unsqueeze(0).cpu(), model.sample_rate)
print("Audio saved to test.wav - listen to check quality!")