
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import warnings
warnings.filterwarnings('ignore')

print('🎤 CSM Ultra-Natural Sofia - Fixed Format...')

try:
    # Load CSM (we know this works now!)
    processor = AutoProcessor.from_pretrained('sesame/csm-1b')
    model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)
    
    print('✅ CSM loaded! Generating ultra-natural speech...')
    
    # Fixed conversation format - use integer speaker IDs
    conversation = [[{
        'role': '0',  # Use '0' instead of 'sofia'
        'content': [
            {
                'type': 'text', 
                'text': 'Hey there! Welcome to Bella Vista! Oh my gosh, I am absolutely thrilled you called today! You know what? Our food here is just incredible!'
            }
        ]
    }]]
    
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
    
    print('🎤 Generating ultra-natural conversational speech...')
    with torch.no_grad():
        audio = model.generate(**inputs, max_length=300, do_sample=True, temperature=0.8)
    
    # Save audio
    audio_tensor = audio[0].cpu().float()
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    torchaudio.save('csm-sofia-ultra-natural.wav', audio_tensor, 24000)
    print('🎉 SUCCESS! CSM Sofia ultra-natural voice saved!')
    print('🎧 This should sound completely human-like!')
    print('📁 File: csm-sofia-ultra-natural.wav')
    
    # Test multiple phrases
    test_phrases = [
        'Oh wonderful! What can I help you with today? I just love talking about our amazing menu!',
        'Mmm, that sounds perfect! You know, our truffle pasta is made fresh every single morning with the most incredible ingredients.',
        'Bellissimo! That is such a great choice! Would you like me to recommend a beautiful wine to pair with that?'
    ]
    
    print('\n🎭 Creating multiple ultra-natural samples...')
    for i, phrase in enumerate(test_phrases):
        conversation = [[{
            'role': '0',
            'content': [{'type': 'text', 'text': phrase}]
        }]]
        
        inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
        
        with torch.no_grad():
            audio = model.generate(**inputs, max_length=400, do_sample=True, temperature=0.8)
        
        audio_tensor = audio[0].cpu().float()
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        filename = f'csm-sofia-sample-{i+1}.wav'
        torchaudio.save(filename, audio_tensor, 24000)
        print(f'✅ Saved: {filename}')
    
    print('\n🎉 All CSM ultra-natural samples created!')
    print('\n🎉 All CSM ultra-natural samples created!')
    print('🎧 These should sound completely human with natural expressions!')

except Exception as e:
    print(f'❌ CSM error: {e}')
    import traceback
    traceback.print_exc()
