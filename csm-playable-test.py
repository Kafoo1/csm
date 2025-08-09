import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import numpy as np
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

print('🎤 CSM Sofia - MP3 Format (Windows Compatible)...')

try:
    # Load CSM
    processor = AutoProcessor.from_pretrained('sesame/csm-1b')
    model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)
    
    print('✅ CSM loaded!')
    
    # Test phrases
    test_phrases = [
        'Welcome to Bella Vista! I am absolutely delighted you called today!',
        'Oh wonderful! What can I help you with? Our menu is fantastic!',
        'Bellissimo! That sounds perfect! You will love our truffle pasta!'
    ]
    
    for i, phrase in enumerate(test_phrases):
        print(f'🎭 Generating sample {i+1}: "{phrase[:30]}..."')
        
        conversation = [[{
            'role': '0',
            'content': [{'type': 'text', 'text': phrase}]
        }]]
        
        inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
        
        with torch.no_grad():
            audio = model.generate(input_ids=inputs, max_new_tokens=100, do_sample=True, temperature=0.8)
        
        # Convert to numpy array
        audio_tensor = audio[0].cpu().float().numpy()
        
        # Normalize audio (important for compatibility)
        audio_normalized = audio_tensor / np.max(np.abs(audio_tensor))
        
        # Save as WAV first with proper format
        wav_filename = f'csm-sofia-{i+1}.wav'
        sf.write(wav_filename, audio_normalized, 24000, format='WAV', subtype='PCM_16')
        
        print(f'✅ WAV saved: {wav_filename}')
        
        # Convert to MP3 using torchaudio
        try:
            audio_tensor_torch = torch.from_numpy(audio_normalized).unsqueeze(0)
            mp3_filename = f'csm-sofia-{i+1}.mp3'
            torchaudio.save(mp3_filename, audio_tensor_torch, 24000, format="mp3")
            print(f'🎵 MP3 saved: {mp3_filename}')
        except Exception as mp3_error:
            print(f'❌ MP3 conversion failed: {mp3_error}')
            print('💡 WAV file should still work')
    
    print('\n🎉 CSM Sofia audio files created!')
    print('🎧 Try playing both WAV and MP3 files')
    print('📁 Files created in current directory')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()