import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import torchaudio
import warnings
warnings.filterwarnings('ignore')

class SofiaVoiceAI:
    def __init__(self):
        self.model_name = 'sesame/csm-1b'
        self.processor = None
        self.model = None
        self.sample_rate = 24000
        
    def initialize(self):
        '''Initialize CSM model for ultra-natural speech'''
        try:
            print('🎤 Initializing Sofia\'s ultra-natural voice...')
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map='cpu'
            )
            
            print('✅ Sofia\'s voice AI ready!')
            return True
            
        except Exception as e:
            print(f'❌ CSM initialization failed: {e}')
            return False
    
    def generate_speech(self, text, speaker='sofia'):
        '''Generate ultra-natural speech from text'''
        try:
            print(f'🎤 Sofia speaking: "{text[:50]}..."')
            
            # Create conversation format
            conversation = [
                [
                    {
                        'role': speaker,
                        'content': [
                            {
                                'type': 'text', 
                                'text': text
                            }
                        ]
                    }
                ]
            ]
            
            # Process input
            inputs = self.processor.apply_chat_template(
                conversation, 
                tokenize=True, 
                return_tensors='pt'
            )
            
            # Generate speech
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs, 
                    max_length=500,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9
                )
            
            # Process audio output
            audio_tensor = audio_values[0].cpu()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            return audio_tensor
            
        except Exception as e:
            print(f'❌ Speech generation failed: {e}')
            return None
    
    def save_audio(self, audio_tensor, filename):
        '''Save audio tensor to file'''
        try:
            torchaudio.save(filename, audio_tensor, self.sample_rate)
            print(f'💾 Audio saved: {filename}')
            return True
        except Exception as e:
            print(f'❌ Save failed: {e}')
            return False

# Test Sofia's voice
if __name__ == '__main__':
    print('🎭 Creating Sofia - Ultra-Natural Restaurant AI Voice')
    
    sofia = SofiaVoiceAI()
    
    if sofia.initialize():
        # Test different restaurant scenarios
        test_phrases = [
            'Welcome to Bella Vista! How can I help you today?',
            'Our special today is homemade truffle pasta with fresh herbs.',
            'That sounds like a wonderful choice! Would you like to add a glass of Chianti?',
            'Perfect! Your order will be ready in about twenty minutes.'
        ]
        
        for i, phrase in enumerate(test_phrases):
            print(f'\n🎤 Generating Sofia voice sample {i+1}...')
            audio = sofia.generate_speech(phrase)
            
            if audio is not None:
                filename = f'sofia_sample_{i+1}.wav'
                sofia.save_audio(audio, filename)
            else:
                print(f'❌ Failed to generate sample {i+1}')
        
        print('\n🎉 Sofia\'s voice samples complete!')
        print('🎧 Play the .wav files to hear your ultra-natural AI assistant!')
    else:
        print('❌ Sofia initialization failed')
print('❌ Sofia initialization failed')