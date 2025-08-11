# save as test_current.py
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except:
    print("❌ PyTorch not installed")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except:
    print("❌ Transformers not installed")

try:
    import torchaudio
    print(f"✅ Torchaudio: {torchaudio.__version__}")
except:
    print("❌ Torchaudio not installed")

try:
    import sentencepiece
    print("✅ Sentencepiece installed")
except:
    print("❌ Sentencepiece NOT installed (this is the problem)")

# Test if we can load models without sentencepiece
print("\nTesting model loading without sentencepiece...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("✅ Can load models without sentencepiece!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")