
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import warnings
warnings.filterwarnings('ignore')

print('🎤 Loading Sofia with CPU optimizations...')

try:
    # Load with CPU optimizations
    model_name = 'sesame/csm-1b'
    
    print('🔄 Loading processor...')
    processor = AutoProcessor.from_pretrained(model_name)
    print('✅ Processor loaded')
    
    print('🔄 Loading model with CPU settings...')
    model = CsmForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True,
        offload_folder='./offload'
    )
    print('✅ Sofia model loaded successfully!')
    
    # Quick test
    conversation = [[{
        'role': 'sofia',
        'content': [{'type': 'text', 'text': 'Hello! Welcome to Bella Vista!'}]
    }]]
    
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
    
    print('🎤 Generating quick test...')
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_length=100)
    
    audio_tensor = audio_values[0].cpu()
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    torchaudio.save('sofia_test.wav', audio_tensor, 24000)
    print('🎉 SUCCESS! Sofia voice saved as sofia_test.wav')
    
except Exception as e:
    print('❌ CSM loading failed:', str(e))
    print('💡 Let\'s use OpenAI TTS while we optimize CSM')
