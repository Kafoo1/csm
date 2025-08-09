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
        """Generate natural speech like Maya"""
        
        # Maya-style conversation setup
        conversation = []
        
        # Add context history if provided (this makes speech more natural)
        if context_history:
            for ctx in context_history:
                conversation.append({
                    "role": str(ctx.get('speaker', 0)), 
                    "content": [{"type": "text", "text": ctx['text']}]
                })
        
        # Add current text with proper speaker formatting
        conversation.append({
            "role": str(speaker_id), 
            "content": [{"type": "text", "text": text}]
        })
        
        # Process input
        inputs = self.processor.apply_chat_template(
            conversation, 
            tokenize=True, 
            return_dict=True,
        ).to(self.device)
        
        # MAYA-STYLE GENERATION PARAMETERS
        with torch.no_grad():
            audio = self.model.generate(
                **inputs,
                output_audio=True,
                
                # Speed and naturalness settings
                max_new_tokens=120,        # Good length for natural speech
                min_new_tokens=20,         # Prevent too short responses
                
                # Natural variation (like Maya)
                do_sample=True,            # Enable natural variation
                temperature=0.8,           # Natural variation (not too random)
                top_p=0.9,                 # Nucleus sampling for naturalness
                
                # Speech quality settings
                repetition_penalty=1.1,    # Prevent repetitive sounds
                length_penalty=1.0,        # Natural length
                
                # Decoder settings for better audio quality
                depth_decoder_do_sample=True,
                depth_decoder_temperature=0.7,  # Natural audio variation
            )
        
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

# TEST MAYA-STYLE GENERATION
def test_maya_quality():
    """Test different scenarios like Maya"""
    
    print('🧪 Testing Maya-style CSM quality...')
    agent = MayaRestaurantAgent()
    
    test_scenarios = [
        {
            "name": "greeting",
            "text": "Welcome to Bella Vista! How can I help you today?",
            "context": []
        },
        {
            "name": "menu_inquiry", 
            "text": "We have amazing fresh pasta and wood-fired pizza today!",
            "context": [{"speaker": 1, "text": "What's good on the menu?"}]
        },
        {
            "name": "order_confirmation",
            "text": "Perfect! One margherita pizza and a house salad. That'll be ready in fifteen minutes.",
            "context": [
                {"speaker": 1, "text": "I'd like a margherita pizza"},
                {"speaker": 0, "text": "Great choice! Anything else?"},
                {"speaker": 1, "text": "Maybe a house salad too"}
            ]
        },
        {
            "name": "friendly_closing",
            "text": "Thank you so much! We'll have that ready for you soon. Have a wonderful day!",
            "context": [{"speaker": 1, "text": "That sounds perfect, thank you"}]
        }
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f'\n🎯 Test {i+1}: {scenario["name"]}')
        print(f'📝 Text: "{scenario["text"]}"')
        
        start_time = time.time()
        
        # Generate with context for naturalness
        audio = agent.csm.create_natural_speech(
            text=scenario["text"],
            speaker_id=0,
            context_history=scenario["context"]
        )
        
        generation_time = time.time() - start_time
        
        # Save with descriptive name
        filename = f"maya_style_{scenario['name']}.wav"
        agent.csm.save_natural_audio(audio, filename)
        
        print(f'⚡ Generated in {generation_time:.2f} seconds')
        print(f'🎧 Saved: {filename}')
    
    print('\n🎉 Maya-style testing complete!')
    print('📋 Files created:')
    print('   - maya_style_greeting.wav')
    print('   - maya_style_menu_inquiry.wav') 
    print('   - maya_style_order_confirmation.wav')
    print('   - maya_style_friendly_closing.wav')
    
    print('\n🎯 Expected improvements:')
    print('   ✅ Natural speech rhythm (no more "ciaooooo")')
    print('   ✅ Clean audio (no background noise)')
    print('   ✅ Appropriate emotions (no evil laugh)')
    print('   ✅ Faster, more natural pace')
    print('   ✅ Contextual awareness')

if __name__ == "__main__":
    # Test the Maya-style implementation
    test_maya_quality()
    
    print('\n🚀 Next steps after testing:')
    print('1. Test the audio files - they should sound like Maya')
    print('2. If quality is good, integrate with Node.js service')
    print('3. Connect to Bird.com for SMS/WhatsApp')
    print('4. Deploy complete voice restaurant system')