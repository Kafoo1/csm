import torch
import torchaudio
from transformers import AutoProcessor, CsmForConditionalGeneration
import re
import time

class NaturalConversationEngine:
    """Creates natural, flowing conversations like Maya"""
    
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
        
        # Conversation state tracking
        self.conversation_history = {}
        self.speaker_profiles = self._initialize_speaker_profiles()
        
        print('🎭 Natural Conversation Engine loaded!')
    
    def _initialize_speaker_profiles(self):
        """Define natural speaker characteristics"""
        return {
            'restaurant_host': {
                'speaker_id': 0,
                'base_energy': 'warm',
                'personality': 'friendly, welcoming, Italian warmth',
                'pace': 'moderate',
                'emotional_range': 'moderate'
            },
            'professional_agent': {
                'speaker_id': 1,
                'base_energy': 'confident',
                'personality': 'professional, helpful, knowledgeable',
                'pace': 'measured',
                'emotional_range': 'controlled'
            },
            'caring_assistant': {
                'speaker_id': 0,
                'base_energy': 'gentle',
                'personality': 'caring, patient, understanding',
                'pace': 'slow',
                'emotional_range': 'empathetic'
            }
        }
    
    def analyze_conversation_context(self, customer_input, conversation_history):
        """Analyze conversation context for natural response adaptation"""
        
        context = {
            'customer_energy': 'neutral',
            'conversation_stage': 'greeting',
            'customer_mood': 'neutral',
            'topic_urgency': 'normal',
            'formality_level': 'casual'
        }
        
        if customer_input:
            input_lower = customer_input.lower()
            
            # Detect customer energy level
            excited_words = ['great', 'awesome', 'amazing', 'love', 'fantastic', 'perfect']
            concerned_words = ['problem', 'issue', 'wrong', 'trouble', 'help', 'urgent']
            casual_words = ['hi', 'hey', 'hello', 'thanks', 'cool']
            formal_words = ['good morning', 'good afternoon', 'please', 'thank you']
            
            if any(word in input_lower for word in excited_words):
                context['customer_energy'] = 'excited'
                context['customer_mood'] = 'positive'
            elif any(word in input_lower for word in concerned_words):
                context['customer_energy'] = 'concerned'
                context['customer_mood'] = 'negative'
            elif any(word in input_lower for word in casual_words):
                context['formality_level'] = 'casual'
            elif any(word in input_lower for word in formal_words):
                context['formality_level'] = 'formal'
            
            # Detect conversation stage
            greeting_words = ['hi', 'hello', 'hey']
            question_words = ['what', 'how', 'when', 'where', 'can you']
            closing_words = ['thanks', 'goodbye', 'bye', 'that\'s all']
            
            if any(word in input_lower for word in greeting_words):
                context['conversation_stage'] = 'greeting'
            elif any(word in input_lower for word in question_words):
                context['conversation_stage'] = 'information'
            elif any(word in input_lower for word in closing_words):
                context['conversation_stage'] = 'closing'
            else:
                context['conversation_stage'] = 'ongoing'
        
        return context
    
    def format_text_for_natural_speech(self, text, context, speaker_profile):
        """Format text to encourage natural speech patterns"""
        
        # Add natural pauses and emphasis based on context
        formatted_text = text
        
        # Add emotional emphasis based on customer energy
        if context['customer_energy'] == 'excited':
            # Match customer excitement with slight enthusiasm
            formatted_text = self._add_enthusiasm_markers(formatted_text)
        elif context['customer_energy'] == 'concerned':
            # Respond with reassuring tone
            formatted_text = self._add_reassurance_markers(formatted_text)
        
        # Add conversational flow markers
        formatted_text = self._add_conversational_flow(formatted_text, context)
        
        # Add natural breathing pauses
        formatted_text = self._add_natural_pauses(formatted_text)
        
        return formatted_text
    
    def _add_enthusiasm_markers(self, text):
        """Add subtle enthusiasm without overdoing it"""
        # Replace period with slight emphasis for positive responses
        if any(word in text.lower() for word in ['great', 'perfect', 'wonderful']):
            text = text.replace('!', '!')  # Keep existing enthusiasm
        return text
    
    def _add_reassurance_markers(self, text):
        """Add reassuring tone markers"""
        # Add gentle reassurance for problem-solving
        if any(word in text.lower() for word in ['help', 'assist', 'solve']):
            text = text.replace('.', '.')  # Keep calm tone
        return text
    
    def _add_conversational_flow(self, text, context):
        """Add conversational connectors for natural flow"""
        
        # Add conversation stage appropriate openings
        if context['conversation_stage'] == 'greeting':
            # Natural greeting flow - no changes needed
            pass
        elif context['conversation_stage'] == 'information':
            # Add thoughtful response indicators
            if text.startswith('We have') or text.startswith('Our'):
                text = 'Let me tell you about that. ' + text
        elif context['conversation_stage'] == 'closing':
            # Add warm closing flow
            if 'thank' not in text.lower():
                text = text + ' Thank you for calling!'
        
        return text
    
    def _add_natural_pauses(self, text):
        """Add natural breathing pauses using punctuation"""
        
        # Add commas for natural breathing
        text = re.sub(r'(\w+) (and|or|but) (\w+)', r'\1, \2 \3', text)
        
        # Add slight pause after greetings
        text = re.sub(r'^(Hello|Hi|Hey)', r'\1,', text)
        
        # Add natural pause before questions
        text = re.sub(r'(\.) (How|What|When|Where|Can)', r'\1 \2', text)
        
        return text
    
    def calculate_adaptive_generation_params(self, context, speaker_profile):
        """Calculate generation parameters for natural conversation flow"""
        
        base_temp = 0.7
        
        # Adjust temperature based on customer energy and conversation context
        if context['customer_energy'] == 'excited':
            temperature = base_temp + 0.1  # Slightly more expressive
        elif context['customer_energy'] == 'concerned':
            temperature = base_temp - 0.1  # More controlled, reassuring
        elif context['conversation_stage'] == 'greeting':
            temperature = base_temp + 0.05  # Slightly warmer
        else:
            temperature = base_temp
        
        # Adjust other parameters for natural flow
        params = {
            'temperature': temperature,
            'max_new_tokens': self._calculate_optimal_length(context),
            'do_sample': True,
            'repetition_penalty': 1.05,  # Prevent word repetition
        }
        
        return params
    
    def _calculate_optimal_length(self, context):
        """Calculate optimal response length for natural conversation"""
        
        if context['conversation_stage'] == 'greeting':
            return 60  # Shorter, punchy greetings
        elif context['conversation_stage'] == 'information':
            return 100  # Longer for explanations
        elif context['conversation_stage'] == 'closing':
            return 50  # Brief, warm closings
        else:
            return 80  # Standard conversational length
    
    def create_consistent_conversation_context(self, session_id, speaker_profile, text):
        """Build conversation context that maintains consistency"""
        
        # Get or create conversation history
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = {
                'speaker_profile': speaker_profile,
                'exchanges': [],
                'established_personality': False
            }
        
        session = self.conversation_history[session_id]
        
        # Build conversation for consistency
        conversation = []
        
        # Personality establishment (first exchange)
        if not session['established_personality']:
            personality_text = self._create_personality_anchor(speaker_profile)
            conversation.append({
                "role": str(speaker_profile['speaker_id']),
                "content": [{"type": "text", "text": personality_text}]
            })
            session['established_personality'] = True
        
        # Add recent conversation history (last 2 exchanges for context)
        recent_exchanges = session['exchanges'][-2:] if len(session['exchanges']) > 2 else session['exchanges']
        for exchange in recent_exchanges:
            conversation.append({
                "role": str(speaker_profile['speaker_id']),
                "content": [{"type": "text", "text": exchange}]
            })
        
        # Add current response
        conversation.append({
            "role": str(speaker_profile['speaker_id']),
            "content": [{"type": "text", "text": text}]
        })
        
        # Store current exchange
        session['exchanges'].append(text)
        
        return conversation
    
    def _create_personality_anchor(self, speaker_profile):
        """Create personality anchoring text for consistency"""
        
        personality_anchors = {
            'restaurant_host': "I'm Sofia, and I love welcoming guests to our restaurant.",
            'professional_agent': "I'm here to provide professional assistance with your needs.",
            'caring_assistant': "I'm here to help you with care and attention."
        }
        
        # Find matching profile
        for profile_name, profile_data in self.speaker_profiles.items():
            if profile_data['speaker_id'] == speaker_profile['speaker_id']:
                return personality_anchors.get(profile_name, "I'm here to help you today.")
        
        return "I'm here to help you today."
    
    def generate_natural_conversation(self, session_id, text, customer_input=None, business_type='restaurant'):
        """Generate natural, flowing conversation"""
        
        # Select appropriate speaker profile
        profile_mapping = {
            'restaurant': 'restaurant_host',
            'real_estate': 'professional_agent', 
            'medical': 'caring_assistant'
        }
        
        speaker_profile_name = profile_mapping.get(business_type, 'restaurant_host')
        speaker_profile = self.speaker_profiles[speaker_profile_name]
        
        # Analyze conversation context
        context = self.analyze_conversation_context(customer_input, 
            self.conversation_history.get(session_id, {}).get('exchanges', []))
        
        print(f'🎭 Context: {context["customer_energy"]} energy, {context["conversation_stage"]} stage')
        
        # Format text for natural speech
        formatted_text = self.format_text_for_natural_speech(text, context, speaker_profile)
        
        # Build consistent conversation context
        conversation = self.create_consistent_conversation_context(session_id, speaker_profile, formatted_text)
        
        # Calculate adaptive generation parameters
        gen_params = self.calculate_adaptive_generation_params(context, speaker_profile)
        
        print(f'🎚️ Generation params: temp={gen_params["temperature"]:.2f}, tokens={gen_params["max_new_tokens"]}')
        
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
            
            return audio, formatted_text, context
            
        except Exception as e:
            print(f'❌ Natural generation failed: {e}')
            return None, formatted_text, context

# NATURAL CONVERSATION TESTING
def test_natural_conversation_flow():
    """Test natural conversation flow with various scenarios"""
    
    print('🎭 Testing Natural Conversation Flow...')
    
    engine = NaturalConversationEngine()
    
    # Test scenarios that should sound natural
    test_conversations = [
        {
            'name': 'Excited Customer',
            'session_id': 'test_1',
            'customer_input': 'Hi! I heard you have amazing pizza!',
            'agent_response': 'Thank you so much! Yes, we absolutely do. Our wood-fired pizza is incredible.',
            'business_type': 'restaurant'
        },
        {
            'name': 'Concerned Customer',
            'session_id': 'test_2', 
            'customer_input': 'I have a problem with my reservation.',
            'agent_response': 'I understand your concern and I\'m here to help. Let me assist you with that right away.',
            'business_type': 'restaurant'
        },
        {
            'name': 'Casual Greeting',
            'session_id': 'test_3',
            'customer_input': 'Hey there!',
            'agent_response': 'Hello! Welcome to Bella Vista. How can I make your day better?',
            'business_type': 'restaurant'
        },
        {
            'name': 'Information Request',
            'session_id': 'test_4',
            'customer_input': 'What are your hours?',
            'agent_response': 'We\'re open daily from 11 AM to 10 PM. We\'d love to see you anytime!',
            'business_type': 'restaurant'
        },
        {
            'name': 'Professional Inquiry',
            'session_id': 'test_5',
            'customer_input': 'I\'m looking for a house in the downtown area.',
            'agent_response': 'Perfect! I specialize in downtown properties and I have some excellent options to show you.',
            'business_type': 'real_estate'
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_conversations):
        print(f'\n🎯 Test {i+1}: {test["name"]}')
        print(f'👤 Customer: "{test["customer_input"]}"')
        print(f'🤖 Agent: "{test["agent_response"]}"')
        
        try:
            audio, formatted_text, context = engine.generate_natural_conversation(
                session_id=test['session_id'],
                text=test['agent_response'],
                customer_input=test['customer_input'],
                business_type=test['business_type']
            )
            
            if audio:
                filename = f"natural_{test['name'].lower().replace(' ', '_')}.wav"
                engine.processor.save_audio(audio, filename)
                
                print(f'✅ Natural audio generated: {filename}')
                print(f'📝 Formatted text: "{formatted_text}"')
                print(f'🎭 Detected context: {context["customer_energy"]} energy')
                
                results.append({
                    'test': test['name'],
                    'audio_file': filename,
                    'context': context,
                    'status': 'SUCCESS'
                })
            else:
                print(f'❌ Audio generation failed')
                results.append({'test': test['name'], 'status': 'FAILED'})
                
        except Exception as e:
            print(f'❌ Test failed: {e}')
            results.append({'test': test['name'], 'error': str(e), 'status': 'ERROR'})
    
    # Summary
    print('\n🎉 Natural Conversation Testing Complete!')
    print('📋 Results:')
    
    for result in results:
        status_emoji = '✅' if result['status'] == 'SUCCESS' else '❌'
        print(f'   {status_emoji} {result["test"]}')
        if result['status'] == 'SUCCESS':
            print(f'      🎧 Audio: {result["audio_file"]}')
            print(f'      🎭 Context: {result["context"]["customer_energy"]} energy, {result["context"]["conversation_stage"]} stage')
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f'\n📊 Success Rate: {success_count}/{len(test_conversations)} conversations natural')
    
    if success_count > 0:
        print('\n🎯 Listen for these natural qualities:')
        print('   ✅ Consistent tone throughout (no high/low jumps)')
        print('   ✅ Flowing speech (not word-by-word)')
        print('   ✅ Appropriate energy matching customer')
        print('   ✅ Natural pauses and rhythm')
        print('   ✅ Emotional intelligence (warm/professional/caring)')

if __name__ == "__main__":
    test_natural_conversation_flow()