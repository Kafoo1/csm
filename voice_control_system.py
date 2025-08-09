import torch
import torchaudio
from transformers import AutoProcessor, CsmForConditionalGeneration
import json
import time

# 1. VOICE CONTROL SYSTEM
class VoiceController:
    """Controls CSM voice selection and consistency"""
    
    # Available voices in CSM (discovered through testing)
    VOICE_PROFILES = {
        'maya_female': {
            'speaker_id': 0,
            'description': 'Natural female voice, warm and professional',
            'personality': 'friendly, helpful, empathetic'
        },
        'professional_male': {
            'speaker_id': 1, 
            'description': 'Professional male voice, confident',
            'personality': 'authoritative, clear, direct'
        },
        'friendly_female': {
            'speaker_id': 2,
            'description': 'Younger female voice, energetic',
            'personality': 'enthusiastic, casual, upbeat'
        },
        'calm_male': {
            'speaker_id': 3,
            'description': 'Calm male voice, reassuring', 
            'personality': 'patient, understanding, gentle'
        }
    }
    
    @classmethod
    def get_voice_for_business(cls, business_type):
        """Select appropriate voice for business type"""
        voice_mapping = {
            'restaurant': 'maya_female',
            'medical': 'calm_male', 
            'real_estate': 'professional_male',
            'retail': 'friendly_female',
            'default': 'maya_female'
        }
        return voice_mapping.get(business_type, 'maya_female')

# 2. TONE ADAPTATION SYSTEM (Like Maya)
class ToneAdaptationEngine:
    """Adapts voice tone based on conversation context"""
    
    TONE_KEYWORDS = {
        'excited': ['great', 'amazing', 'wonderful', 'fantastic', 'love'],
        'concerned': ['problem', 'issue', 'wrong', 'error', 'help'],
        'professional': ['business', 'meeting', 'appointment', 'schedule'],
        'friendly': ['hello', 'hi', 'thanks', 'please', 'welcome'],
        'urgent': ['emergency', 'urgent', 'asap', 'immediately', 'quickly']
    }
    
    TONE_TEMPERATURES = {
        'excited': 0.9,      # More expressive
        'concerned': 0.6,    # More controlled  
        'professional': 0.5, # Very controlled
        'friendly': 0.8,     # Natural warmth
        'urgent': 0.7        # Clear but urgent
    }
    
    def detect_tone(self, conversation_text, customer_input):
        """Detect appropriate tone from conversation context"""
        text = f"{conversation_text} {customer_input}".lower()
        
        tone_scores = {}
        for tone, keywords in self.TONE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            tone_scores[tone] = score
        
        # Return tone with highest score, default to friendly
        detected_tone = max(tone_scores, key=tone_scores.get) if max(tone_scores.values()) > 0 else 'friendly'
        return detected_tone, self.TONE_TEMPERATURES[detected_tone]

# 3. BUSINESS KNOWLEDGE BASE SYSTEM
class BusinessKnowledgeBase:
    """Multi-tenant knowledge base for different businesses"""
    
    def __init__(self):
        self.businesses = {}
    
    def add_business(self, business_id, config):
        """Add business configuration and knowledge"""
        self.businesses[business_id] = {
            'name': config['name'],
            'type': config['type'],  # restaurant, medical, retail, etc.
            'phone_number': config['phone_number'],
            'voice_profile': VoiceController.get_voice_for_business(config['type']),
            'knowledge': config['knowledge'],
            'personality': config.get('personality', {}),
            'responses': config.get('responses', {}),
            'created_at': time.time()
        }
        
    def get_business_context(self, phone_number=None, business_id=None):
        """Get business context from phone number or ID"""
        if phone_number:
            for bid, biz in self.businesses.items():
                if biz['phone_number'] == phone_number:
                    return bid, biz
        elif business_id and business_id in self.businesses:
            return business_id, self.businesses[business_id]
        
        return None, None

# 4. UNIFIED VOICE AGENT (Multi-Tenant)
class UniversalVoiceAgent:
    """One agent that can serve multiple businesses"""
    
    def __init__(self):
        # Load CSM once for all businesses
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sesame/csm-1b"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()
        
        # Initialize systems
        self.knowledge_base = BusinessKnowledgeBase()
        self.tone_engine = ToneAdaptationEngine()
        
        # Session tracking
        self.active_sessions = {}  # phone_number -> business context
        
        print('🚀 Universal Voice Agent loaded!')
    
    def handle_incoming_call(self, phone_number, customer_speech=None):
        """Main entry point when Bird.com forwards a call"""
        
        # 1. Identify which business this call is for
        business_id, business_config = self.knowledge_base.get_business_context(phone_number=phone_number)
        
        if not business_config:
            return self._handle_unknown_business(phone_number)
        
        # 2. Set up session context
        self.active_sessions[phone_number] = {
            'business_id': business_id,
            'business_config': business_config,
            'conversation_history': [],
            'voice_profile': business_config['voice_profile']
        }
        
        print(f'📞 Call received for {business_config["name"]} ({business_config["type"]})')
        print(f'🎤 Using voice: {business_config["voice_profile"]}')
        
        # 3. Generate appropriate response
        if customer_speech:
            return self.respond_to_customer(phone_number, customer_speech)
        else:
            # Generate greeting
            greeting = business_config['responses'].get('greeting', 
                f"Hello! Thank you for calling {business_config['name']}. How can I help you today?")
            return self.generate_voice_response(phone_number, greeting)
    
    def respond_to_customer(self, phone_number, customer_speech):
        """Generate contextual response to customer"""
        
        session = self.active_sessions.get(phone_number)
        if not session:
            return None
        
        business_config = session['business_config']
        
        # 1. Add customer speech to history
        session['conversation_history'].append({
            'role': 'customer',
            'text': customer_speech
        })
        
        # 2. Generate intelligent response using business knowledge
        response_text = self._generate_intelligent_response(customer_speech, business_config)
        
        # 3. Add response to history
        session['conversation_history'].append({
            'role': 'agent', 
            'text': response_text
        })
        
        # 4. Generate voice with appropriate tone
        return self.generate_voice_response(phone_number, response_text)
    
    def generate_voice_response(self, phone_number, response_text):
        """Generate voice with business-specific voice and adaptive tone"""
        
        session = self.active_sessions[phone_number]
        business_config = session['business_config']
        voice_profile = VoiceController.VOICE_PROFILES[session['voice_profile']]
        
        # 1. Detect appropriate tone
        conversation_text = ' '.join([h['text'] for h in session['conversation_history'][-3:]])
        tone, temperature = self.tone_engine.detect_tone(conversation_text, response_text)
        
        print(f'🎭 Detected tone: {tone} (temp: {temperature})')
        
        # 2. Build conversation with consistent speaker ID
        conversation = []
        
        # Add business context for consistency
        conversation.append({
            "role": str(voice_profile['speaker_id']),
            "content": [{"type": "text", "text": f"I work at {business_config['name']}."}]
        })
        
        # Add recent conversation history for context (but keep same speaker)
        for exchange in session['conversation_history'][-2:]:
            if exchange['role'] == 'agent':
                conversation.append({
                    "role": str(voice_profile['speaker_id']),
                    "content": [{"type": "text", "text": exchange['text'][:100]}]
                })
        
        # Add current response
        conversation.append({
            "role": str(voice_profile['speaker_id']),
            "content": [{"type": "text", "text": response_text}]
        })
        
        # 3. Generate with tone-adapted parameters
        try:
            inputs = self.processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True
            ).to(self.device)
            
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=temperature,  # Adaptive tone
                )
            
            return audio, response_text
            
        except Exception as e:
            print(f'❌ Voice generation failed: {e}')
            return None, response_text
    
    def _generate_intelligent_response(self, customer_speech, business_config):
        """Generate intelligent response using business knowledge"""
        
        # This is where you'd integrate with OpenAI GPT-4 or similar
        # For now, simple pattern matching
        
        speech_lower = customer_speech.lower()
        business_type = business_config['type']
        business_name = business_config['name']
        
        # Business-specific responses
        if business_type == 'restaurant':
            if any(word in speech_lower for word in ['menu', 'food', 'eat', 'order']):
                return f"Great! Let me tell you about our menu at {business_name}. We have amazing fresh pasta, wood-fired pizza, and daily specials."
            elif any(word in speech_lower for word in ['reservation', 'table', 'book']):
                return f"I'd be happy to help you make a reservation at {business_name}. What day and time work best for you?"
        
        elif business_type == 'real_estate':
            if any(word in speech_lower for word in ['house', 'property', 'buy', 'sell']):
                return f"Excellent! I'm here to help with all your real estate needs. Are you looking to buy or sell a property?"
        
        # Default response
        return f"Thank you for calling {business_name}. How can I assist you today?"
    
    def _handle_unknown_business(self, phone_number):
        """Handle calls to unregistered numbers"""
        return None, "I'm sorry, but this number is not configured in our system."

# 5. BUSINESS SETUP EXAMPLES
def setup_sample_businesses():
    """Set up sample businesses for testing"""
    
    agent = UniversalVoiceAgent()
    
    # Restaurant setup
    agent.knowledge_base.add_business('bella_vista', {
        'name': 'Bella Vista Restaurant',
        'type': 'restaurant',
        'phone_number': '+1234567890',
        'knowledge': {
            'menu': ['margherita pizza', 'carbonara pasta', 'tiramisu'],
            'hours': '11am - 10pm',
            'location': '123 Main Street'
        },
        'personality': {
            'style': 'warm, Italian hospitality',
            'tone': 'friendly and welcoming'
        },
        'responses': {
            'greeting': 'Ciao! Welcome to Bella Vista! How can I help you today?'
        }
    })
    
    # Real Estate setup  
    agent.knowledge_base.add_business('dream_homes', {
        'name': 'Dream Homes Realty',
        'type': 'real_estate', 
        'phone_number': '+1234567891',
        'knowledge': {
            'services': ['buying', 'selling', 'market analysis'],
            'areas': ['downtown', 'suburbs', 'waterfront'],
            'agents': ['Sarah Johnson', 'Mike Chen']
        },
        'personality': {
            'style': 'professional, knowledgeable',
            'tone': 'confident and trustworthy'
        },
        'responses': {
            'greeting': 'Hello! Thank you for calling Dream Homes Realty. How can I help you with your real estate needs?'
        }
    })
    
    return agent

# 6. TESTING THE SYSTEM
def test_multi_tenant_system():
    """Test the multi-tenant voice agent"""
    
    print('🧪 Testing Universal Voice Agent...')
    
    # Set up businesses
    agent = setup_sample_businesses()
    
    # Test restaurant call
    print('\n🍕 Testing restaurant call...')
    audio, text = agent.handle_incoming_call('+1234567890')
    if audio:
        agent.processor.save_audio(audio, 'restaurant_greeting.wav')
        print(f'✅ Restaurant: {text}')
    
    # Test customer interaction
    print('\n👤 Customer asks about menu...')
    audio, text = agent.respond_to_customer('+1234567890', "What's on your menu today?")
    if audio:
        agent.processor.save_audio(audio, 'restaurant_menu_response.wav')
        print(f'✅ Menu response: {text}')
    
    # Test real estate call
    print('\n🏠 Testing real estate call...')
    audio, text = agent.handle_incoming_call('+1234567891')
    if audio:
        agent.processor.save_audio(audio, 'realestate_greeting.wav')
        print(f'✅ Real Estate: {text}')
    
    print('\n🎉 Multi-tenant testing complete!')

if __name__ == "__main__":
    test_multi_tenant_system()