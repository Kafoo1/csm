print("🎤 Testing CSM setup...")

try:
    from transformers import AutoProcessor, CsmForConditionalGeneration
    print("✅ CSM transformers imports successful")
    
    # Test model loading (this will tell us if we need Hugging Face access)
    print("🔄 Attempting to load CSM model...")
    model_name = "sesame/csm-1b"
    processor = AutoProcessor.from_pretrained(model_name)
    print("✅ CSM model accessible!")
    
except Exception as e:
    print("❌ CSM test failed:", str(e))
    print("💡 You may need Hugging Face access to sesame/csm-1b")

print("🎯 CSM basic test complete!")