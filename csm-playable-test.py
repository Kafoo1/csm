
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import subprocess
import os

print('🎤 CSM Sofia - Windows-Compatible Format...')

# Load and generate (same as before)
processor = AutoProcessor.from_pretrained('sesame/csm-1b')
model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)

conversation = [[{
    'role': '0',
    'content': [{'type': 'text', 'text': 'Welcome to Bella Vista! I am absolutely delighted you called today!'}]
}]]

inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')

with torch.no_grad():
    audio = model.generate(input_ids=inputs, max_new_tokens=100)

audio_tensor = audio[0].cpu().float()
if len(audio_tensor.shape) == 1:
    audio_tensor = audio_tensor.unsqueeze(0)

# Save as WAV first
wav_file = 'csm-sofia-temp.wav'
torchaudio.save(wav_file, audio_tensor, 22050)  # Standard sample rate

print(f'✅ WAV saved: {wav_file}')
print(f'📊 File size: {os.path.getsize(wav_file)} bytes')

# Try to play with Windows default player
try:
    import winsound
    print('🎧 Playing with Windows sound...')
    winsound.PlaySound(wav_file, winsound.SND_FILENAME)
    print('✅ Playback successful!')
except Exception as e:
    print(f'❌ Windows playback failed: {e}')
    print('💡 Try opening the WAV file manually with Windows Media Player')

print(f'📁 File location: {os.path.abspath(wav_file)}')
