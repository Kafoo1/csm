# Test if everything works with Python 3.11
import sys
print(f"Python: {sys.version}")

# Test imports
try:
    import sentencepiece
    print("✅ Sentencepiece loaded!")
except:
    print("❌ Sentencepiece failed")

try:
    from transformers import CsmForConditionalGeneration, AutoProcessor
    print("✅ Transformers CSM support loaded!")
    
    # Load the model
    print("\nLoading CSM-1B model...")
    model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b")
    processor = AutoProcessor.from_pretrained("sesame/csm-1b")
    print("✅ CSM Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Test the generator from CSM repo
try:
    from generator import load_csm_1b
    model = load_csm_1b(device="cpu")
    print("✅ CSM generator loaded!")
except Exception as e:
    print(f"⚠️ Generator not available: {e}")