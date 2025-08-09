import torch
import torchaudio
from transformers import AutoProcessor, CsmForConditionalGeneration
import re
import time

class SimpleNaturalVoice:
    """Simple but natural voice generation that actually works with CSM"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sesame/csm-1b"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()
        print('🎭 Simple Natural Voice loaded!')
    
    def make_text_natural(self, text, customer_energy="neutral", business_type="restaurant"):
        """Make text more natural and conversational"""
        
        # Add personality based on business type
        if business_type == "restaurant":
            personality_prefix = ""  # Keep it simple, let the voice handle personality
        elif business_type == "real_estate":
            personality_prefix = ""
        else:
            personality_prefix = ""
        
        # Add natural conversation flow
        natural_text = self._add_natural_flow(text, customer_energy)
        
        # Add appropriate emotion without overdoing it
        natural_text = self._adjust_for_energy(natural_text, customer_energy)
        
        return personality_prefix + natural_text
    
    def _add_natural_flow(self, text, customer_energy):
        """Add natural conversation connectors"""
        
        # Add natural openings for different types of responses
        if text.startswith("We have") or text.startswith("Our"):
            if customer_energy == "excited":
                text = "Oh wonderful! " + text
            elif customer_energy == "concerned":
                text = "Absolutely, " + text
            else:
                text = "Yes, " + text
        
        # Add natural pauses with commas
        text = re.sub(r' and ', ', and ', text)
        text = re.sub(r' but ', ', but ', text)
        
        # Add natural ending flow
        if not text.endswith(('!', '?', '.')):
            text += "."
        
        return text
    
    def _adjust_for_energy(self, text, customer_energy):
        """Subtly adjust text for customer energy without overdoing it"""
        
        if customer_energy == "excited":
            # Add slight warmth, but don't overdo it
            if "!" not in text and any(word in text.lower() for word in ["amazing", "great", "wonderful", "perfect"]):
                text = text.replace(".", "!")
        
        elif customer_energy == "concerned":
            # Add reassurance
            if text.startswith("I "):
                pass  # Keep as is - already reassuring
            elif any(word in text.lower() for word in ["help", "assist"]):
                if not text.startswith("I"):
                    text = "I'll " + text.lower()
        
        return text
    
    def detect_customer_energy(self, customer_input):
        """Simple energy detection"""
        if not customer_input:
            return "neutral"
        
        input_lower = customer_input.lower()
        
        # Excited indicators
        excited_words = ['amazing', 'awesome', 'great', 'love', 'fantastic', '!']
        if any(word in input_lower for word in excited_words):
            return "excited"
        
        # Concerned indicators  
        concerned_words = ['problem', 'issue', 'wrong', 'help', 'trouble']
        if any(word in input_lower for word in concerned_words):
            return "concerned"
        
        return "neutral"
    
    def get_natural_generation_params(self, customer_energy, business_type):
        """Get optimal parameters for natural speech"""
        
        # Base parameters that work well
        base_params = {
            'max_new_tokens': 80,
            'do_sample': True,
            'temperature': 0.75,  # Good middle ground
        }
        
        # Adjust slightly based on context
        if customer_energy == "excited":
            base_params['temperature'] = 0.8  # Slightly more expressive
        elif customer_energy == "concerned":
            base_params['temperature'] = 0.7  # More controlled
            base_params['max_new_tokens'] = 90  # Slightly longer for reassurance
        
        # Business type adjustments
        if business_type == "real_estate":
            base_params['temperature'] = 0.72  # Professional but warm
        
        return base_params
    
    def generate_natural_speech(self, text, customer_input=None, business_type="restaurant"):
        """Generate natural speech with simple, working approach"""
        
        # Detect customer energy
        customer_energy = self.detect_customer_energy(customer_input)
        print(f'🎭 Detected energy: {customer_energy}')
        
        # Make text more natural
        natural_text = self.make_text_natural(text, customer_energy, business_type)
        print(f'📝 Natural text: "{natural_text}"')
        
        # Get speaker ID for business type
        speaker_id = 0 if business_type == "restaurant" else (1 if business_type == "real_estate" else 0)
        
        # SIMPLE CONVERSATION FORMAT (that actually works)
        conversation = [
            {"role": str(speaker_id), "content": [{"type": "text", "text": natural_text}]}
        ]
        
        # Get natural generation parameters
        gen_params = self.get_natural_generation_params(customer_energy, business_type)
        print(f'🎚️ Params: temp={gen_params["temperature"]}, tokens={gen_params["max_new_tokens"]}')
        
        try:
            # Process input
            inputs = self.processor.apply_chat_template(
                conversation, 
                tokenize=True, 
                return_dict=True,
            ).to(self.device)
            
            # Generate with natural parameters
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    **gen_params
                )
            
            return audio, natural_text, customer_energy
            
        except Exception as e:
            print(f'❌ Generation failed: {e}')
            return None, natural_text, customer_energy

# CLEAN AUDIO POST-PROCESSING
class SimpleAudioCleaner:
    """Simple audio cleaning that preserves naturalness"""
    
    def clean_audio_preserve_naturalness(self, audio_file, output_file):
        """Clean audio while preserving natural speech qualities"""
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Very gentle cleaning to preserve naturalness
            cleaned = self._gentle_cleaning(waveform, sample_rate)
            
            # Save
            torchaudio.save(output_file, cleaned, sample_rate)
            print(f'✅ Gently cleaned: {output_file}')
            return output_file
            
        except Exception as e:
            print(f'❌ Cleaning failed: {e}')
            return audio_file
    
    def _gentle_cleaning(self, waveform, sample_rate):
        """Very gentle cleaning that preserves speech naturalness"""
        
        # 1. Gentle normalization (preserve dynamic range)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.88  # Not too loud
        
        # 2. Very light noise gate (only remove obvious silence)
        noise_threshold = 0.01  # Very low threshold
        mask = torch.abs(waveform) > noise_threshold
        
        # Smooth the mask to avoid cutting off natural speech
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        mask_float = mask.float().unsqueeze(1)
        mask_padded = torch.nn.functional.pad(mask_float, (2, 2), mode='reflect')
        smoothed_mask = torch.nn.functional.conv1d(mask_padded, kernel, padding=0).squeeze(1)
        
        # Apply smooth mask
        waveform = waveform * smoothed_mask
        
        # 3. Final gentle normalization
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.85
        
        return waveform

# COMPREHENSIVE TESTING
def test_simple_natural_voice():
    """Test simple natural voice generation"""
    
    print('🎭 Testing Simple Natural Voice Generation...')
    
    voice_engine = SimpleNaturalVoice()
    audio_cleaner = SimpleAudioCleaner()
    
    # Test scenarios focusing on natural flow
    test_scenarios = [
        {
            'name': 'excited_response',
            'text': 'Thank you so much! Yes, we absolutely do. Our wood-fired pizza is incredible.',
            'customer_input': 'Hi! I heard you have amazing pizza!',
            'business_type': 'restaurant'
        },
        {
            'name': 'concerned_response',
            'text': 'I understand your concern and I\'m here to help. Let me assist you with that right away.',
            'customer_input': 'I have a problem with my reservation.',
            'business_type': 'restaurant'
        },
        {
            'name': 'casual_greeting',
            'text': 'Hello! Welcome to Bella Vista. How can I make your day better?',
            'customer_input': 'Hey there!',
            'business_type': 'restaurant'
        },
        {
            'name': 'professional_response',
            'text': 'Perfect! I specialize in downtown properties and I have some excellent options to show you.',
            'customer_input': 'I\'m looking for a house in the downtown area.',
            'business_type': 'real_estate'
        },
        {
            'name': 'menu_explanation',
            'text': 'We have fresh pasta, wood-fired pizza, and amazing daily specials. What sounds good to you?',
            'customer_input': 'What do you recommend?',
            'business_type': 'restaurant'
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_scenarios):
        print(f'\n🎯 Test {i+1}: {test["name"]}')
        print(f'👤 Customer: "{test["customer_input"]}"')
        
        try:
            # Generate natural speech
            audio, natural_text, energy = voice_engine.generate_natural_speech(
                text=test['text'],
                customer_input=test['customer_input'],
                business_type=test['business_type']
            )
            
            if audio:
                # Save original
                original_file = f"simple_{test['name']}_original.wav"
                voice_engine.processor.save_audio(audio, original_file)
                
                # Create cleaned version
                clean_file = f"simple_{test['name']}_clean.wav"
                audio_cleaner.clean_audio_preserve_naturalness(original_file, clean_file)
                
                print(f'✅ Generated: {original_file} and {clean_file}')
                print(f'📝 Natural text: "{natural_text}"')
                print(f'⚡ Energy detected: {energy}')
                
                results.append({
                    'test': test['name'],
                    'original_file': original_file,
                    'clean_file': clean_file,
                    'energy': energy,
                    'status': 'SUCCESS'
                })
            else:
                print(f'❌ Audio generation failed')
                results.append({'test': test['name'], 'status': 'FAILED'})
                
        except Exception as e:
            print(f'❌ Test failed: {e}')
            results.append({'test': test['name'], 'error': str(e), 'status': 'ERROR'})
    
    # Summary
    print('\n🎉 Simple Natural Voice Testing Complete!')
    print('📋 Results:')
    
    success_count = 0
    for result in results:
        if result['status'] == 'SUCCESS':
            success_count += 1
            print(f'   ✅ {result["test"]} - {result["energy"]} energy')
            print(f'      🎧 Original: {result["original_file"]}')
            print(f'      🧹 Clean: {result["clean_file"]}')
        else:
            print(f'   ❌ {result["test"]} - {result.get("error", "Failed")}')
    
    print(f'\n📊 Success Rate: {success_count}/{len(test_scenarios)} tests passed')
    
    if success_count > 0:
        print('\n🎯 What to listen for:')
        print('   ✅ Consistent voice tone (no random jumps)')
        print('   ✅ Natural conversational flow (not robotic)')
        print('   ✅ Appropriate energy matching customer input')
        print('   ✅ Clean audio without artifacts')
        print('   ✅ Natural pauses and rhythm')
        
        print('\n🚀 If these sound natural:')
        print('   1. Integrate with Bird.com for real calls')
        print('   2. Add business knowledge bases')
        print('   3. Deploy complete voice agent system')

if __name__ == "__main__":
    test_simple_natural_voice()