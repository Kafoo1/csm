import torch
import torchaudio
import os
import time
from transformers import AutoProcessor, CsmForConditionalGeneration

print('🎤 Maya-Style CSM - Natural Voice Generation...')

class MayaStyleCSM:
    def __init__(self):
        """Initialize CSM with Maya-like settings"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'🖥️ Using device: {self.device}')
        
        # Load model with optimizations
        model_id = "sesame/csm-1b"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Set to eval mode for consistent outputs
        self.model.eval()
        print('✅ Maya-style CSM loaded!')
    
    def create_natural_speech(self, text, speaker_id=0, context_history=None):
        """Generate natural speech like Maya with safe parameters"""
        
        try:
            # Maya-style conversation setup
            conversation = []
            
            # Add context history if provided (but limit to prevent token overflow)
            if context_history:
                # Limit context to last 2 exchanges to prevent token issues
                recent_context = context_history[-2:] if len(context_history) > 2 else context_history
                for ctx in recent_context:
                    conversation.append({
                        "role": str(ctx.get('speaker', 0)), 
                        "content": [{"type": "text", "text": ctx['text'][:100]}]  # Limit text length
                    })
            
            # Add current text with proper speaker formatting (limit length for safety)
            safe_text = text[:150]  # Prevent overly long inputs
            conversation.append({
                "role": str(speaker_id), 
                "content": [{"type": "text", "text": safe_text}]
            })
            
            # Process input with error handling
            inputs = self.processor.apply_chat_template(
                conversation, 
                tokenize=True, 
                return_dict=True,
            ).to(self.device)
            
            # Check input token length for safety
            if hasattr(inputs, 'input_ids') and inputs.input_ids.shape[1] > 500:
                print('⚠️ Input too long, truncating...')
                inputs.input_ids = inputs.input_ids[:, :500]
            
            # SAFE GENERATION PARAMETERS (Fixed token bounds)
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    
                    # Safe length settings
                    max_new_tokens=100,        # Safe token limit
                    
                    # Conservative sampling (prevents token errors)
                    do_sample=True,            # Enable natural variation
                    temperature=0.7,           # Safe temperature range
                    
                    # Remove problematic parameters that cause token errors
                    # top_p=0.9,               # REMOVED - causes vocab issues
                    # repetition_penalty=1.1,  # REMOVED - causes token bounds error
                    # length_penalty=1.0,      # REMOVED - not needed
                    # depth_decoder_do_sample=True,        # REMOVED - not supported
                    # depth_decoder_temperature=0.7,      # REMOVED - causes error
                )
            
            return audio
            
        except Exception as e:
            print(f'❌ Generation failed: {e}')
            print('🔄 Trying with minimal parameters...')
            
            # Fallback: Use absolute minimal parameters
            simple_conversation = [{"role": str(speaker_id), "content": [{"type": "text", "text": safe_text}]}]
            inputs = self.processor.apply_chat_template(simple_conversation, tokenize=True, return_dict=True).to(self.device)
            
            with torch.no_grad():
                audio = self.model.generate(**inputs, output_audio=True, max_new_tokens=50)
            
            return audio
    
    def save_natural_audio(self, audio, filename, clean_audio=True):
        """Save audio with Maya-style post-processing"""
        
        # Save the audio using processor (official method)
        self.processor.save_audio(audio, filename)
        
        if clean_audio:
            # Post-process for cleaner audio (like Maya)
            self.clean_audio_file(filename)
        
        print(f'✅ Natural audio saved: {filename}')
        return filename
    
    def clean_audio_file(self, filename):
        """Clean audio to remove artifacts (Maya-style)"""
        try:
            # Load the audio
            waveform, sample_rate = torchaudio.load(filename)
            
            # Apply gentle normalization (prevent clipping)
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val * 0.95  # Slight headroom
            
            # Simple noise gate (remove very quiet background noise)
            noise_threshold = 0.01
            waveform = torch.where(torch.abs(waveform) < noise_threshold, 
                                 torch.zeros_like(waveform), 
                                 waveform)
            
            # Save cleaned audio
            torchaudio.save(filename, waveform, sample_rate)
            
        except Exception as e:
            print(f'⚠️ Audio cleaning failed: {e}')

# MAYA-STYLE RESTAURANT AGENT
class MayaRestaurantAgent:
    def __init__(self):
        self.csm = MayaStyleCSM()
        self.conversation_history = []
        
        # Maya's personality context
        self.personality_context = [
            {"speaker": 0, "text": "Hello, I'm Sofia from Bella Vista restaurant."},
            {"speaker": 0, "text": "I love helping customers find the perfect meal."}
        ]
    
    def respond(self, customer_input=None, response_text=None):
        """Generate Maya-style response"""
        
        # Add customer input to history
        if customer_input:
            self.conversation_history.append({"speaker": 1, "text": customer_input})
        
        # Use provided response or generate one
        if not response_text:
            response_text = "Welcome to Bella Vista! How can I help you today?"
        
        # Generate natural speech with context
        full_context = self.personality_context + self.conversation_history[-3:]  # Last 3 exchanges
        
        audio = self.csm.create_natural_speech(
            text=response_text,
            speaker_id=0,  # Sofia's voice
            context_history=full_context
        )
        
        # Add to history
        self.conversation_history.append({"speaker": 0, "text": response_text})
        
        return audio

# TEST MAYA-STYLE GENERATION (Simplified and Safe)
def test_maya_quality():
    """Test different scenarios like Maya with safe parameters"""
    
    print('🧪 Testing Maya-style CSM quality (safe mode)...')
    
    try:
        agent = MayaRestaurantAgent()
        
        # Simplified test scenarios (shorter text to prevent token issues)
        test_scenarios = [
            {
                "name": "greeting",
                "text": "Welcome to Bella Vista! How can I help you?",
                "context": []
            },
            {
                "name": "menu", 
                "text": "We have fresh pasta and pizza today!",
                "context": []  # Remove context initially for simplicity
            },
            {
                "name": "order",
                "text": "Perfect! One pizza coming up.",
                "context": []
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f'\n🎯 Test {i+1}: {scenario["name"]}')
            print(f'📝 Text: "{scenario["text"]}"')
            
            try:
                start_time = time.time()
                
                # Generate with simplified parameters
                audio = agent.csm.create_natural_speech(
                    text=scenario["text"],
                    speaker_id=0,
                    context_history=scenario["context"]
                )
                
                generation_time = time.time() - start_time
                
                # Save with descriptive name
                filename = f"maya_safe_{scenario['name']}.wav"
                agent.csm.save_natural_audio(audio, filename, clean_audio=False)  # Skip cleaning initially
                
                print(f'⚡ Generated in {generation_time:.2f} seconds')
                print(f'✅ Saved: {filename}')
                
            except Exception as e:
                print(f'❌ Test {i+1} failed: {e}')
                print('🔄 Continuing with next test...')
                continue
        
        print('\n🎉 Safe testing complete!')
        print('📋 Check these files:')
        print('   - maya_safe_greeting.wav')
        print('   - maya_safe_menu.wav') 
        print('   - maya_safe_order.wav')
        
    except Exception as e:
        print(f'❌ Test setup failed: {e}')
        print('🔄 Trying basic generation...')
        
        # Absolute minimal test
        try:
            csm = MayaStyleCSM()
            audio = csm.create_natural_speech("Hello from Sesame", speaker_id=0)
            csm.save_natural_audio(audio, "basic_test.wav", clean_audio=False)
            print('✅ Basic test worked: basic_test.wav')
        except Exception as e2:
            print(f'❌ Even basic test failed: {e2}')
            print('💡 Try the original working method instead')

# ALTERNATIVE: Super Simple Test
def simple_test():
    """Absolute minimal test using official method"""
    print('🧪 Running super simple test...')
    
    try:
        from transformers import CsmForConditionalGeneration, AutoProcessor
        
        model_id = "sesame/csm-1b"
        device = "cpu"  # Force CPU for stability
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
        
        # Minimal conversation
        conversation = [{"role": "0", "content": [{"type": "text", "text": "Hello from Bella Vista!"}]}]
        inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(device)
        
        # Minimal generation
        audio = model.generate(**inputs, output_audio=True)
        processor.save_audio(audio, "simple_working.wav")
        
        print('✅ Simple test SUCCESS: simple_working.wav')
        return True
        
    except Exception as e:
        print(f'❌ Simple test failed: {e}')
        return False

if __name__ == "__main__":
    print('🚀 Starting CSM Audio Quality Tests...')
    
    # Try the improved Maya-style first
    print('\n📋 Option 1: Testing Maya-style implementation...')
    try:
        test_maya_quality()
    except Exception as e:
        print(f'❌ Maya-style test failed: {e}')
        
        # Fallback to simple test
        print('\n📋 Option 2: Trying simple fallback...')
        success = simple_test()
        
        if not success:
            print('\n📋 Option 3: Manual debug suggestions...')
            print('💡 Try these commands manually:')
            print('   python -c "from transformers import CsmForConditionalGeneration; print(\'✅ Transformers works\')"')
            print('   python -c "import torch; print(\'✅ PyTorch works\')"')
    
    print('\n🎯 Next steps:')
    print('1. Test any generated .wav files')
    print('2. If audio sounds good, integrate with voice agent')
    print('3. Connect to Bird.com for SMS/WhatsApp')
    print('4. Deploy complete restaurant system')