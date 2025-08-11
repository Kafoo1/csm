import torch
from transformers import AutoModel, AutoTokenizer
import requests
import io

# Download the actual model weights
model_id = "sesame/csm-1b"

print("Downloading CSM model directly...")
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,  # IMPORTANT: This allows custom code
    revision="main"
)

# The key is accessing the audio generation component
if hasattr(model, 'generate_audio'):
    audio = model.generate_audio("Hello, this is a test.")
    print("✅ Audio generated!")
else:
    print("Model components:", dir(model))
    print("Looking for audio generation method...")