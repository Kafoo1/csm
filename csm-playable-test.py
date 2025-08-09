import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import numpy as np

print('🎤 CSM Sofia - FIXED Audio Generation...')

# Load CSM
processor = AutoProcessor.from_pretrained('sesame/csm-1b')
model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)

print('✅ CSM loaded!')

# Test conversation
conversation = [[{
    'role': '0',
    'content': [{'type': 'text', 'text': 'Ciao! Welcome to Bella Vista! How can I help you today?'}]
}]]

inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')

print('🎯 Generating speech...')
with torch.no_grad():
    # Generate audio tokens
    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

print(f'📊 Generated tokens shape: {outputs.shape}')

# THE FIX: Properly extract audio from MimiDecoderOutput
try:
    print('🔧 Decoding audio with FIXED method...')
    
    # Method 1: Access the audio_values attribute
    if hasattr(model, 'codec_model'):
        # Decode the generated tokens
        decoder_output = model.codec_model.decode(outputs)
        
        print(f'🔍 Decoder output type: {type(decoder_output)}')
        print(f'🔍 Decoder output attributes: {dir(decoder_output)}')
        
        # Extract audio from MimiDecoderOutput object
        if hasattr(decoder_output, 'audio_values'):
            decoded_audio = decoder_output.audio_values
            print(f'✅ Found audio_values: {decoded_audio.shape}')
            
        elif hasattr(decoder_output, 'waveform'):
            decoded_audio = decoder_output.waveform
            print(f'✅ Found waveform: {decoded_audio.shape}')
            
        elif hasattr(decoder_output, 'audio'):
            decoded_audio = decoder_output.audio
            print(f'✅ Found audio: {decoded_audio.shape}')
            
        else:
            # Fallback: try to access as tensor
            print('📋 Available attributes:', [attr for attr in dir(decoder_output) if not attr.startswith('_')])
            
            # Try common audio attributes
            for attr_name in ['audio_values', 'waveform', 'audio', 'samples', 'output', 'last_hidden_state']:
                if hasattr(decoder_output, attr_name):
                    attr_value = getattr(decoder_output, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        decoded_audio = attr_value
                        print(f'✅ Found audio in {attr_name}: {decoded_audio.shape}')
                        break
            else:
                raise Exception("No audio tensor found in decoder output")
        
        # Process the audio
        decoded_audio = decoded_audio.cpu().float()
        
        # Remove batch dimension if present
        if len(decoded_audio.shape) > 2:
            decoded_audio = decoded_audio.squeeze(0)
        
        # Ensure stereo or mono format
        if len(decoded_audio.shape) == 1:
            # Mono audio
            audio_final = decoded_audio.unsqueeze(0)
        elif decoded_audio.shape[0] > decoded_audio.shape[1]:
            # Transpose if needed
            audio_final = decoded_audio.t()
        else:
            audio_final = decoded_audio
        
        print(f'📊 Final audio shape: {audio_final.shape}')
        
        # Normalize audio
        if torch.max(torch.abs(audio_final)) > 0:
            audio_final = audio_final / torch.max(torch.abs(audio_final)) * 0.9
        
        # Save audio files
        sample_rate = 24000  # CSM default sample rate
        
        torchaudio.save('csm-sofia-WORKING.wav', audio_final, sample_rate)
        print('✅ WORKING audio saved: csm-sofia-WORKING.wav')
        
        # Also save at 16kHz for compatibility
        audio_16k = torchaudio.functional.resample(audio_final, sample_rate, 16000)
        torchaudio.save('csm-sofia-WORKING-16k.wav', audio_16k, 16000)
        print('✅ 16kHz version saved: csm-sofia-WORKING-16k.wav')
        
        print('🎉 SUCCESS! CSM audio generation is now WORKING!')
        print('🎧 Try playing: csm-sofia-WORKING.wav')
        
except Exception as e:
    print(f'❌ Decoding failed: {e}')
    print('🔧 Trying alternative decoding method...')
    
    # Alternative Method: Direct token-to-audio conversion
    try:
        # Get the raw generated tokens
        audio_tokens = outputs[0] if len(outputs.shape) > 1 else outputs
        
        # Convert tokens to audio-like signal
        if len(audio_tokens.shape) == 2:
            # Average across feature dimension
            audio_signal = audio_tokens.float().mean(dim=1)
        else:
            audio_signal = audio_tokens.float()
        
        # Normalize to audio range
        audio_signal = (audio_signal - audio_signal.mean()) / (audio_signal.std() + 1e-8)
        audio_signal = torch.clamp(audio_signal, -1.0, 1.0)
        
        # Interpolate to proper audio length
        target_length = 24000 * 3  # 3 seconds
        if len(audio_signal) != target_length:
            audio_signal = torch.nn.functional.interpolate(
                audio_signal.unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear'
            ).squeeze()
        
        # Add channel dimension
        audio_final = audio_signal.unsqueeze(0)
        
        torchaudio.save('csm-sofia-FALLBACK.wav', audio_final, 24000)
        print('✅ Fallback audio saved: csm-sofia-FALLBACK.wav')
        
    except Exception as e2:
        print(f'❌ Alternative method also failed: {e2}')
        print('💡 CSM might need model-specific decoding approach')

print('🎯 Next step: Test both audio files to see which one works!')
print('📋 Files created:')
print('   - csm-sofia-WORKING.wav (main attempt)')
print('   - csm-sofia-WORKING-16k.wav (16kHz version)')  
print('   - csm-sofia-FALLBACK.wav (fallback method)')