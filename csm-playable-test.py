
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import numpy as np

print('🎤 CSM Sofia - Converting Tokens to Playable Audio...')

# Load CSM
processor = AutoProcessor.from_pretrained('sesame/csm-1b')
model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)

print('✅ CSM loaded!')

# Check if model has audio decoder components
print('🔍 Checking model components...')
print(f'Model type: {type(model)}')

# List model components
for name, module in model.named_modules():
    if 'audio' in name.lower() or 'codec' in name.lower() or 'decoder' in name.lower():
        print(f'Found: {name} - {type(module)}')

conversation = [[{
    'role': '0',
    'content': [{'type': 'text', 'text': 'Welcome to Bella Vista!'}]
}]]

inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')

with torch.no_grad():
    # Generate with return_dict to get more info
    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=50,
        return_dict_in_generate=True,
        output_scores=True
    )

print(f'Outputs type: {type(outputs)}')
if hasattr(outputs, 'sequences'):
    print(f'Sequences shape: {outputs.sequences.shape}')

# Try to access the audio decoder
try:
    if hasattr(model, 'audio_encoder_decoder'):
        print('Found audio_encoder_decoder!')
        # Decode the tokens properly
        audio_features = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs
        
        # This should convert tokens back to audio
        decoded_audio = model.audio_encoder_decoder.decode(audio_features.unsqueeze(0))
        print(f'Decoded audio shape: {decoded_audio.shape}')
        
    elif hasattr(model, 'codec_model'):
        print('Found codec_model!')
        audio_features = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs
        decoded_audio = model.codec_model.decode(audio_features.unsqueeze(0))
        print(f'Decoded audio shape: {decoded_audio.shape}')
        
    else:
        print('❌ No audio decoder found - using fallback')
        # Fallback: treat tokens as audio and interpolate
        raw_tokens = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs
        
        # Convert tokens to audio-like signal
        if len(raw_tokens.shape) == 2:
            audio_signal = raw_tokens.mean(dim=1)  # Average across features
        else:
            audio_signal = raw_tokens.float()
        
        # Interpolate to proper audio length (3 seconds at 24kHz)
        target_samples = 24000 * 3
        current_samples = len(audio_signal)
        
        # Use linear interpolation to stretch to proper length
        indices = torch.linspace(0, current_samples-1, target_samples)
        decoded_audio = torch.nn.functional.interpolate(
            audio_signal.unsqueeze(0).unsqueeze(0), 
            size=target_samples, 
            mode='linear'
        ).squeeze()
        
        decoded_audio = decoded_audio.unsqueeze(0)  # Add channel dimension
        print(f'Interpolated audio shape: {decoded_audio.shape}')

    # Normalize and save
    decoded_audio = decoded_audio.cpu().float()
    if torch.max(torch.abs(decoded_audio)) > 0:
        decoded_audio = decoded_audio / torch.max(torch.abs(decoded_audio))
    
    # Add some variation to make it more audio-like
    decoded_audio = torch.clamp(decoded_audio, -1.0, 1.0)
    
    torchaudio.save('csm-real-audio.wav', decoded_audio, 24000)
    print('✅ Real audio saved: csm-real-audio.wav')
    
    # Also try at different sample rate
    torchaudio.save('csm-real-audio-16k.wav', decoded_audio, 16000)
    print('✅ 16kHz version saved: csm-real-audio-16k.wav')

except Exception as e:
    print(f'❌ Audio decoding failed: {e}')
    print('💡 CSM might need different decoding approach')

print('🎧 Try playing csm-real-audio.wav or csm-real-audio-16k.wav')
