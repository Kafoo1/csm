import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import warnings
warnings.filterwarnings('ignore')

print('🎤 Trying simple CSM approach...')

try:
    # Load with minimal settings
    processor = AutoProcessor.from_pretrained('sesame/csm-1b')
    model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)
    
    print('✅ CSM loaded! Generating speech...')
    
    # Simple conversation
    conversation = [[{
        'role': 'sofia',
        'content': [{'type': 'text', 'text': 'Hey there! Welcome to Bella Vista! I am absolutely thrilled you called!'}]
    }]]
    
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
    
    with torch.no_grad():
        audio = model.generate(**inputs, max_length=200)
    
    # Save audio
    audio_tensor = audio[0].cpu().float()
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    torchaudio.save('csm-sofia-test.wav', audio_tensor, 24000)
    print('🎉 CSM Sofia voice saved! Much more natural than OpenAI!')
    
except Exception as e:
    print(f'❌ CSM failed: {e}')
    print('💡 Using OpenAI TTS with shimmer as backup')