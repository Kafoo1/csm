import torch
import torchaudio
from transformers import AutoProcessor, CsmForConditionalGeneration
import json
import time

# SIMPLE NOISE REDUCTION (No external libraries)
class SimpleNoiseReducer:
    """Simple, reliable noise reduction using only PyTorch/torchaudio"""
    
    def clean_audio_simple(self, audio_file, output_file):
        """Simple but effective noise cleaning"""
        
        print(f'🔧 Cleaning audio: {audio_file}')
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Simple cleaning operations
            cleaned = self._basic_cleaning(waveform, sample_rate)
            
            # Save with proper format
            torchaudio.save(output_file, cleaned, sample_rate)
            print(f'✅ Clean audio saved: {output_file}')
            
            return output_file
            
        except Exception as e:
            print(f'❌ Cleaning failed: {e}')
            return audio_file
    
    def _basic_cleaning(self, waveform, sample_rate):
        """Basic audio cleaning operations"""
        
        # 1. Normalize to prevent clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.9
        
        # 2. Simple noise gate (remove very quiet sounds)
        noise_threshold = 0.015
        mask = torch.abs(waveform) > noise_threshold
        waveform = waveform * mask.float()
        
        # 3. Smooth transitions to avoid clicks
        # Apply very light smoothing
        if waveform.shape[1] > 100:
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            waveform_padded = torch.nn.functional.pad(waveform.unsqueeze(1), (1, 1), mode='reflect')
            waveform = torch.nn.functional.conv1d(waveform_padded, kernel, padding=0).squeeze(1)
        
        # 4. Final normalization
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.85
        
        return waveform

# FIXED CSM CONVERSATION FORMAT
class FixedUniversalAgent:
    """Fixed universal agent with correct CSM conversation format"""
    
    def __init__(self):
        # Load CSM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sesame/csm-1b"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()
        
        # Simple noise reducer
        self.noise_reducer = SimpleNoiseReducer()
        
        # Business configurations
        self.businesses = {}
        
        print('🚀 Fixed Universal Voice Agent loaded!')
    
    def add_business(self, business_id, config):
        """Add business configuration"""
        self.businesses[business_id] = config
        print(f'✅ Added business: {config["name"]}')
    
    def get_business_by_phone(self, phone_number):
        """Find business by phone number"""
        for business_id, config in self.businesses.items():
            if config['phone_number'] == phone_number:
                return business_id, config
        return None, None
    
    def generate_clean_speech(self, text, speaker_id=0, business_name=""):
        """Generate speech with correct CSM format and clean audio"""
        
        try:
            # CORRECT CSM FORMAT: Simple conversation (no mixed audio/text)
            conversation = [
                {"role": str(speaker_id), "content": [{"type": "text", "text": text}]}
            ]
            
            # Process input
            inputs = self.processor.apply_chat_template(
                conversation, 
                tokenize=True, 
                return_dict=True,
            ).to(self.device)
            
            # Generate with safe parameters
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=80,   # Shorter for reliability
                    do_sample=True,
                    temperature=0.7,     # Natural but controlled
                )
            
            return audio
            
        except Exception as e:
            print(f'❌ Generation failed: {e}')
            return None
    
    def handle_call(self, phone_number, customer_input=None):
        """Handle incoming call with clean audio generation"""
        
        # Find business
        business_id, business_config = self.get_business_by_phone(phone_number)
        
        if not business_config:
            print(f'❌ Unknown phone number: {phone_number}')
            return None, "Business not found"
        
        print(f'📞 Call for: {business_config["name"]} ({business_config["type"]})')
        
        # Determine response
        if customer_input:
            response_text = self._generate_response(customer_input, business_config)
        else:
            response_text = business_config.get('greeting', 
                f"Hello! Thank you for calling {business_config['name']}. How can I help you?")
        
        # Get appropriate voice
        speaker_id = self._get_speaker_id(business_config['type'])
        print(f'🎤 Using speaker ID: {speaker_id}')
        
        # Generate audio
        audio = self.generate_clean_speech(response_text, speaker_id, business_config['name'])
        
        if audio:
            # Save original
            filename = f"{business_id}_response_{int(time.time())}"
            original_file = f"{filename}.wav"
            self.processor.save_audio(audio, original_file)
            
            # Create cleaned version
            clean_file = f"{filename}_clean.wav"
            self.noise_reducer.clean_audio_simple(original_file, clean_file)
            
            return clean_file, response_text
        
        return None, response_text
    
    def _get_speaker_id(self, business_type):
        """Get appropriate speaker ID for business type"""
        voice_mapping = {
            'restaurant': 0,     # Female voice (Maya-like)
            'real_estate': 1,    # Male professional voice
            'medical': 0,        # Female caring voice
            'retail': 2,         # Friendly female voice
        }
        return voice_mapping.get(business_type, 0)
    
    def _generate_response(self, customer_input, business_config):
        """Generate intelligent response based on business type"""
        
        input_lower = customer_input.lower()
        business_name = business_config['name']
        business_type = business_config['type']
        
        # Restaurant responses
        if business_type == 'restaurant':
            if any(word in input_lower for word in ['menu', 'food', 'order']):
                return f"Great! At {business_name}, we have fresh pasta, wood-fired pizza, and daily specials. What sounds good to you?"
            elif any(word in input_lower for word in ['reservation', 'table']):
                return f"I'd be happy to help with a reservation at {business_name}. What day and time work for you?"
            elif any(word in input_lower for word in ['hours', 'open', 'close']):
                return f"We're open daily from 11 AM to 10 PM. We'd love to see you at {business_name}!"
        
        # Real estate responses
        elif business_type == 'real_estate':
            if any(word in input_lower for word in ['house', 'property', 'buy', 'sell']):
                return f"I'm here to help with all your real estate needs. Are you looking to buy or sell a property?"
            elif any(word in input_lower for word in ['agent', 'realtor']):
                return f"I can connect you with one of our experienced agents at {business_name}. What area are you interested in?"
        
        # Default response
        return f"Thank you for calling {business_name}. How can I assist you today?"

# SETUP TEST BUSINESSES
def setup_test_businesses():
    """Set up test businesses for demonstration"""
    
    agent = FixedUniversalAgent()
    
    # Add restaurant
    agent.add_business('bella_vista', {
        'name': 'Bella Vista Restaurant',
        'type': 'restaurant',
        'phone_number': '+1234567890',
        'greeting': 'Ciao! Welcome to Bella Vista! How can I help you today?'
    })
    
    # Add real estate
    agent.add_business('dream_homes', {
        'name': 'Dream Homes Realty', 
        'type': 'real_estate',
        'phone_number': '+1234567891',
        'greeting': 'Hello! Thank you for calling Dream Homes Realty. How can I help with your real estate needs?'
    })
    
    return agent

# COMPREHENSIVE TESTING
def test_fixed_system():
    """Test the fixed system with clean audio generation"""
    
    print('🧪 Testing Fixed Universal Voice Agent...')
    
    # Setup
    agent = setup_test_businesses()
    
    test_cases = [
        {
            'name': 'Restaurant Greeting',
            'phone': '+1234567890',
            'input': None,
            'expected_voice': 'female'
        },
        {
            'name': 'Restaurant Menu Inquiry',
            'phone': '+1234567890', 
            'input': "What's on your menu today?",
            'expected_voice': 'female'
        },
        {
            'name': 'Real Estate Greeting',
            'phone': '+1234567891',
            'input': None,
            'expected_voice': 'male'
        },
        {
            'name': 'Real Estate Property Question',
            'phone': '+1234567891',
            'input': "I'm looking to buy a house",
            'expected_voice': 'male'
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases):
        print(f'\n🎯 Test {i+1}: {test["name"]}')
        
        try:
            audio_file, response_text = agent.handle_call(test['phone'], test['input'])
            
            if audio_file:
                print(f'✅ Generated: {audio_file}')
                print(f'📝 Response: {response_text}')
                results.append({
                    'test': test['name'],
                    'audio_file': audio_file,
                    'response': response_text,
                    'status': 'SUCCESS'
                })
            else:
                print(f'❌ Failed to generate audio')
                results.append({
                    'test': test['name'],
                    'status': 'FAILED'
                })
                
        except Exception as e:
            print(f'❌ Test failed: {e}')
            results.append({
                'test': test['name'],
                'error': str(e),
                'status': 'ERROR'
            })
    
    # Summary
    print('\n🎉 Testing Complete!')
    print('📋 Results:')
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    for result in results:
        status_emoji = '✅' if result['status'] == 'SUCCESS' else '❌'
        print(f'   {status_emoji} {result["test"]}')
        if result['status'] == 'SUCCESS':
            print(f'      🎧 Audio: {result["audio_file"]}')
    
    print(f'\n📊 Success Rate: {success_count}/{len(test_cases)} tests passed')
    
    if success_count > 0:
        print('\n🎯 Next Steps:')
        print('1. Test the generated audio files - they should play properly')
        print('2. Listen for voice consistency (no switching)')
        print('3. Check audio quality (clean, no artifacts)')
        print('4. If good, integrate with Bird.com webhooks')

if __name__ == "__main__":
    test_fixed_system()