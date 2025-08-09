import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

print('🎤 CSM Sofia - CORRECT OFFICIAL METHOD...')

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cpu"

print(f'🖥️ Using device: {device}')

# Load CSM the OFFICIAL way
model_id = "sesame/csm-1b"
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

print('✅ CSM loaded with official method!')

# Method 1: Simple text input (OFFICIAL WAY)
print('🎯 Method 1: Simple text with speaker ID...')
try:
    text = "[0]Ciao! Welcome to Bella Vista! How can I help you today?"  # [0] = speaker 0
    inputs = processor(text, add_special_tokens=True).to(device)
    
    # THE KEY: Use output_audio=True
    audio = model.generate(**inputs, output_audio=True)
    
    # Save using processor (OFFICIAL METHOD)
    processor.save_audio(audio, "csm-sofia-OFFICIAL-method1.wav")
    print('✅ SUCCESS! Method 1 audio saved: csm-sofia-OFFICIAL-method1.wav')
    
except Exception as e:
    print(f'❌ Method 1 failed: {e}')

# Method 2: Conversation format (RECOMMENDED WAY)
print('🎯 Method 2: Conversation format...')
try:
    conversation = [
        {"role": "0", "content": [{"type": "text", "text": "Welcome to Bella Vista! How can I help you today?"}]},
    ]
    
    inputs = processor.apply_chat_template(
        conversation, 
        tokenize=True, 
        return_dict=True,
    ).to(device)
    
    # Generate with output_audio=True
    audio = model.generate(**inputs, output_audio=True)
    
    # Save using processor
    processor.save_audio(audio, "csm-sofia-OFFICIAL-method2.wav")
    print('✅ SUCCESS! Method 2 audio saved: csm-sofia-OFFICIAL-method2.wav')
    
except Exception as e:
    print(f'❌ Method 2 failed: {e}')

# Method 3: With generation parameters for better quality
print('🎯 Method 3: With better generation parameters...')
try:
    conversation = [
        {"role": "0", "content": [{"type": "text", "text": "Buongiorno! I'm Sofia from Bella Vista restaurant. What delicious dish can I help you order today?"}]},
    ]
    
    inputs = processor.apply_chat_template(
        conversation, 
        tokenize=True, 
        return_dict=True,
    ).to(device)
    
    # Generate with better parameters
    audio = model.generate(
        **inputs, 
        output_audio=True,
        max_new_tokens=150,  # Longer generation
        temperature=0.7,     # More natural variation
        do_sample=True       # Enable sampling
    )
    
    processor.save_audio(audio, "csm-sofia-OFFICIAL-method3.wav")
    print('✅ SUCCESS! Method 3 audio saved: csm-sofia-OFFICIAL-method3.wav')
    
except Exception as e:
    print(f'❌ Method 3 failed: {e}')

print('\n🎉 CSM OFFICIAL IMPLEMENTATION COMPLETE!')
print('📋 Files created using OFFICIAL method:')
print('   - csm-sofia-OFFICIAL-method1.wav (simple text)')
print('   - csm-sofia-OFFICIAL-method2.wav (conversation format)')  
print('   - csm-sofia-OFFICIAL-method3.wav (enhanced parameters)')
print('\n🎧 Test these audio files - they should work perfectly!')
print('💡 If any work, we can integrate this into your voice agent!')