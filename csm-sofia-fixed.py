@"
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import warnings
warnings.filterwarnings('ignore')

print('🎤 CSM Ultra-Natural Sofia - Correct Input Format...')

try:
    # Load CSM (we know this works!)
    processor = AutoProcessor.from_pretrained('sesame/csm-1b')
    model = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b', torch_dtype=torch.float16)
    
    print('✅ CSM loaded! Generating ultra-natural speech...')
    
    # Fixed conversation format
    conversation = [[{
        'role': '0',
        'content': [
            {
                'type': 'text', 
                'text': 'Hey there! Welcome to Bella Vista! Oh my gosh, I am absolutely thrilled you called today! You know what? Our food here is just incredible!'
            }
        ]
    }]]
    
    # Get inputs and check what we receive
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
    print(f'📊 Input type: {type(inputs)}')
    print(f'📊 Input keys: {inputs.keys() if hasattr(inputs, "keys") else "No keys (is tensor)"}')
    
    print('🎤 Generating ultra-natural conversational speech...')
    
    # Fix the generate call based on input type
    with torch.no_grad():
        if isinstance(inputs, dict):
            # If inputs is a dictionary, use it directly
            audio = model.generate(**inputs, max_length=300, do_sample=True, temperature=0.8)
        else:
            # If inputs is a tensor, wrap it properly
            audio = model.generate(input_ids=inputs, max_length=300, do_sample=True, temperature=0.8)
    
    # Save audio
    audio_tensor = audio[0].cpu().float()
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    torchaudio.save('csm-sofia-ultra-natural.wav', audio_tensor, 24000)
    print('🎉 SUCCESS! CSM Sofia ultra-natural voice saved!')
    print('🎧 This should sound completely human-like!')
    print('📁 File: csm-sofia-ultra-natural.wav')
    
    # Test with simpler phrases to ensure it works
    test_phrases = [
        'Welcome to Bella Vista!',
        'How can I help you today?',
        'Our pasta is absolutely delicious!'
    ]
    
    print('\n🎭 Creating simple ultra-natural samples...')
    for i, phrase in enumerate(test_phrases):
        try:
            conversation = [[{
                'role': '0',
                'content': [{'type': 'text', 'text': phrase}]
            }]]
            
            inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors='pt')
            
            with torch.no_grad():
                if isinstance(inputs, dict):
                    audio = model.generate(**inputs, max_length=200)
                else:
                    audio = model.generate(input_ids=inputs, max_length=200)
            
            audio_tensor = audio[0].cpu().float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            filename = f'csm-sofia-simple-{i+1}.wav'
            torchaudio.save(filename, audio_tensor, 24000)
            print(f'✅ Saved: {filename}')
            
        except Exception as sample_error:
            print(f'❌ Sample {i+1} failed: {sample_error}')
    
    print('\n🎉 CSM ultra-natural samples created!')
    print('🎧 These should sound completely human with natural expressions!')

except Exception as e:
    print(f'❌ CSM error: {e}')
    import traceback
    traceback.print_exc()
"@ | Out-File -FilePath "csm-sofia-working.py" -Encoding utf8

python csm-sofia-working.py