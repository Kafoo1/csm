
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print('🎤 CSM Sofia - Fixed Tensor Dimensions...')

try:
    # Load CSM
    processor = AutoProcessor.from_pretrained('sesame/csm-1b')
    model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)
    
    print('✅ CSM loaded!')
    
    # Simple test
    conversation = [[{
        'role': '0',
        'content': [{'type': 'text', 'text': 'Welcome to Bella Vista! I am absolutely delighted you called today!'}]
    }]]
    
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
    
    with torch.no_grad():
        audio_output = model.generate(input_ids=inputs, max_new_tokens=100)
    
    print(f'📊 Raw audio shape: {audio_output.shape}')
    print(f'📊 Audio type: {type(audio_output)}')
    
    # Fix tensor dimensions properly
    if len(audio_output.shape) == 3:
        # Remove batch dimension: (batch, channels, samples) -> (channels, samples)
        audio_tensor = audio_output[0]
        print(f'📊 After removing batch: {audio_tensor.shape}')
    elif len(audio_output.shape) == 2:
        # Already (channels, samples) or (batch, samples)
        if audio_output.shape[0] == 1:
            audio_tensor = audio_output[0].unsqueeze(0)  # Make it (1, samples)
        else:
            audio_tensor = audio_output
    else:
        # 1D tensor, make it (1, samples)
        audio_tensor = audio_output.unsqueeze(0)
    
    print(f'📊 Final audio shape: {audio_tensor.shape}')
    
    # Convert to float and normalize
    audio_tensor = audio_tensor.cpu().float()
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 0:
        audio_tensor = audio_tensor / max_val
    
    print(f'📊 Normalized shape: {audio_tensor.shape}')
    print(f'📊 Audio range: {audio_tensor.min():.3f} to {audio_tensor.max():.3f}')
    
    # Save as WAV (most compatible)
    wav_filename = 'csm-sofia-working.wav'
    torchaudio.save(wav_filename, audio_tensor, 24000)
    print(f'✅ WAV saved: {wav_filename}')
    
    # Try MP3 with fixed dimensions
    try:
        mp3_filename = 'csm-sofia-working.mp3'
        torchaudio.save(mp3_filename, audio_tensor, 24000, format='mp3')
        print(f'🎵 MP3 saved: {mp3_filename}')
    except Exception as mp3_error:
        print(f'❌ MP3 failed: {mp3_error}')
        print('💡 WAV should work fine though!')
    
    # Try different sample rates for compatibility
    try:
        wav_16k = 'csm-sofia-16khz.wav'
        torchaudio.save(wav_16k, audio_tensor, 16000)
        print(f'✅ 16kHz WAV saved: {wav_16k}')
    except Exception as e:
        print(f'❌ 16kHz failed: {e}')
    
    print('\n🎉 CSM Sofia audio files ready!')
    print('🎧 Try playing the WAV files - they should work now!')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
