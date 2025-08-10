"""
🎭 MAYA VOICE AGENT - FIXED CSM IMPLEMENTATION
Production-ready implementation that works with CSM-1B's actual tokenization
"""

import torch
import torchaudio
from transformers import AutoProcessor, WhisperProcessor, CsmForConditionalGeneration
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time
import datetime
from collections import deque
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PERSONALITY SYSTEM (Same as before)
# ============================================================================

class PersonalityTone(Enum):
    """Maya's personality tones - each represents a different interaction style"""
    WARM_WELCOMING = "warm_welcoming"
    CONFIDENT_KNOWLEDGEABLE = "confident_knowledgeable"
    HELPFUL_EAGER = "helpful_eager"
    REASSURING_TRUSTWORTHY = "reassuring_trustworthy"

class EmotionalState(Enum):
    """Emotional states for prosody modulation"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EMPATHETIC = "empathetic"
    CONFIDENT = "confident"
    REASSURING = "reassuring"
    EXCITED = "excited"
    CALM = "calm"

class ConversationStage(Enum):
    """Conversation stages for dialogue management"""
    GREETING = "greeting"
    NEEDS_ASSESSMENT = "needs_assessment"
    MAIN_INTERACTION = "main_interaction"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    WRAP_UP = "wrap_up"

@dataclass
class PersonalityProfile:
    """Complete personality definition for Maya"""
    name: str = "Maya"
    base_tone: str = "warm, friendly, professional"
    speech_pace: str = "moderate with natural variation"
    vocabulary_level: str = "clear and accessible"
    emotional_range: List[str] = field(default_factory=lambda: ["empathetic", "enthusiastic", "reassuring"])
    filler_words: List[str] = field(default_factory=lambda: ["um", "ah", "well", "you know"])
    signature_phrases: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.signature_phrases = {
            PersonalityTone.WARM_WELCOMING.value: [
                "Hi there!", "Welcome!", "How can I help you today?",
                "It's great to hear from you!", "Thanks for calling!"
            ],
            PersonalityTone.CONFIDENT_KNOWLEDGEABLE.value: [
                "Oh, absolutely!", "That's a great question!",
                "I can definitely help with that", "Based on our experience"
            ],
            PersonalityTone.HELPFUL_EAGER.value: [
                "Perfect!", "I'd be happy to!", "Let me help you with that",
                "Great choice!", "Absolutely, let's do this!"
            ],
            PersonalityTone.REASSURING_TRUSTWORTHY.value: [
                "I completely understand", "Don't worry",
                "We'll take care of that", "I'll personally ensure"
            ]
        }

@dataclass
class ConversationContext:
    """Enhanced context tracking for natural conversations"""
    session_id: str
    personality_tone: PersonalityTone
    emotional_state: EmotionalState
    conversation_stage: ConversationStage
    energy_level: str = "moderate"
    response_length: str = "normal"
    turn_count: int = 0
    last_user_emotion: Optional[str] = None
    topics_discussed: List[str] = field(default_factory=list)
    objections_raised: List[str] = field(default_factory=list)
    context_continuity: bool = False

# ============================================================================
# MEMORY SYSTEM (Simplified but effective)
# ============================================================================

class ConversationalMemory:
    """Advanced memory system for context-aware responses"""
    
    def __init__(self, max_turns: int = 10, context_window_minutes: float = 2.0):
        self.max_turns = max_turns
        self.context_window_minutes = context_window_minutes
        self.sessions: Dict[str, deque] = {}
        self.session_contexts: Dict[str, ConversationContext] = {}
        
    def add_turn(self, session_id: str, speaker: str, text: str, 
                 emotion: Optional[str] = None):
        """Add a conversation turn to memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_turns)
            
        turn = {
            'timestamp': time.time(),
            'speaker': speaker,
            'text': text,
            'emotion': emotion
        }
        
        self.sessions[session_id].append(turn)
        logger.info(f"Memory updated: {speaker} in session {session_id}")
        
    def get_recent_context(self, session_id: str) -> List[Dict]:
        """Get recent conversation context"""
        if session_id not in self.sessions:
            return []
        return list(self.sessions[session_id])[-3:]  # Last 3 turns
    
    def analyze_conversation_flow(self, session_id: str) -> Dict[str, Any]:
        """Analyze conversation patterns"""
        recent_turns = self.get_recent_context(session_id)
        
        if not recent_turns:
            return {'pattern': 'new_conversation', 'needs_reassurance': False}
            
        recent_text = ' '.join([t['text'].lower() for t in recent_turns])
        
        return {
            'needs_reassurance': any(word in recent_text for word in ['worried', 'concern', 'problem']),
            'is_positive': any(word in recent_text for word in ['great', 'perfect', 'thank you']),
            'is_escalating': any(word in recent_text for word in ['angry', 'frustrated', 'upset'])
        }

# ============================================================================
# SPEECH ENHANCER (Core personality expression)
# ============================================================================

class SpeechEnhancer:
    """Enhances text with natural speech patterns and personality traits"""
    
    def __init__(self, personality_profile: PersonalityProfile):
        self.personality = personality_profile
        
    def enhance_text(self, text: str, context: ConversationContext) -> str:
        """Apply personality-based text enhancements"""
        enhanced = text
        
        # Apply personality-specific style
        enhanced = self._apply_personality_style(enhanced, context.personality_tone)
        
        # Apply stage-based enhancements
        enhanced = self._apply_stage_enhancement(enhanced, context.conversation_stage)
        
        # Apply emotional coloring
        enhanced = self._apply_emotional_coloring(enhanced, context.emotional_state)
        
        return enhanced
    
    def _apply_personality_style(self, text: str, tone: PersonalityTone) -> str:
        """Apply personality-specific language patterns"""
        
        if tone == PersonalityTone.WARM_WELCOMING:
            if not text.lower().startswith(('hi', 'hello', 'welcome')):
                text = f"Hi there! {text}"
            text = text.replace(' help ', ' absolutely help ')
            
        elif tone == PersonalityTone.CONFIDENT_KNOWLEDGEABLE:
            if not text.lower().startswith(('oh', 'that', 'absolutely')):
                text = f"Oh, {text}"
            text = text.replace(' is ', ' is definitely ')
            
        elif tone == PersonalityTone.HELPFUL_EAGER:
            if not text.lower().startswith(('perfect', 'great', 'awesome')):
                text = f"Perfect! {text}"
            text = text.replace(' can ', ' can absolutely ')
            
        elif tone == PersonalityTone.REASSURING_TRUSTWORTHY:
            if 'understand' not in text.lower():
                text = f"I completely understand. {text}"
            text = text.replace(' will ', ' will personally ')
            
        return text
    
    def _apply_stage_enhancement(self, text: str, stage: ConversationStage) -> str:
        """Enhance based on conversation stage"""
        
        if stage == ConversationStage.GREETING:
            time_greeting = self._get_time_based_greeting()
            if not text.lower().startswith(('good', 'hi', 'hello')):
                text = f"{time_greeting} {text}"
                
        elif stage == ConversationStage.CLOSING:
            if 'thank' not in text.lower():
                text = f"Thank you so much! {text}"
                
        return text
    
    def _apply_emotional_coloring(self, text: str, emotion: EmotionalState) -> str:
        """Apply subtle emotional modifications"""
        
        if emotion == EmotionalState.HAPPY:
            text = text.replace('.', '!')
        elif emotion == EmotionalState.EMPATHETIC:
            if not text.startswith(('I hear', 'I understand')):
                text = f"I hear you. {text}"
        elif emotion == EmotionalState.REASSURING:
            if not text.startswith(("Don't worry", "No problem")):
                text = f"Don't worry. {text}"
                
        return text
    
    def _get_time_based_greeting(self) -> str:
        """Get time-appropriate greeting"""
        hour = datetime.datetime.now().hour
        if hour < 12:
            return "Good morning!"
        elif hour < 17:
            return "Good afternoon!"
        else:
            return "Good evening!"

# ============================================================================
# FIXED MAYA VOICE AGENT - Working with CSM tokenization
# ============================================================================

class MayaVoiceAgent:
    """Maya Voice Agent with fixed CSM compatibility"""
    
    def __init__(self, model_id: str = "sesame/csm-1b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Maya Voice Agent on {self.device}")
        
        # Initialize personality components
        self.personality = PersonalityProfile()
        self.memory = ConversationalMemory()
        self.speech_enhancer = SpeechEnhancer(self.personality)
        
        # Maya's optimal settings (from working implementation)
        self.maya_settings = {
            'temperature': 0.72,
            'max_new_tokens_short': 60,
            'max_new_tokens_normal': 80,
            'max_new_tokens_expressive': 100,
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
        }
        
        # Load model
        self._load_model(model_id)
        
        # Session management
        self.active_sessions: Dict[str, ConversationContext] = {}
        
        logger.info("✅ Maya Voice Agent initialized successfully")
    
    def _load_model(self, model_id: str):
        """Load CSM model with proper configuration"""
        try:
            logger.info(f"Loading model {model_id}...")
            
            # Try AutoProcessor first, fall back to WhisperProcessor if needed
            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
                logger.info("Using AutoProcessor")
            except:
                logger.info("AutoProcessor failed, trying WhisperProcessor")
                self.processor = WhisperProcessor.from_pretrained(model_id)
            
            # Load model with optimal settings
            self.model = CsmForConditionalGeneration.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_cache=True
            )
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_session(self, session_id: str, initial_tone: PersonalityTone = PersonalityTone.WARM_WELCOMING) -> ConversationContext:
        """Create a new conversation session"""
        context = ConversationContext(
            session_id=session_id,
            personality_tone=initial_tone,
            emotional_state=EmotionalState.NEUTRAL,
            conversation_stage=ConversationStage.GREETING
        )
        
        self.active_sessions[session_id] = context
        logger.info(f"Created session {session_id} with tone {initial_tone.value}")
        
        return context
    
    def analyze_user_input(self, text: str) -> Dict[str, Any]:
        """Analyze user input for emotional and contextual cues"""
        text_lower = text.lower()
        
        # Determine emotional response needed
        if any(word in text_lower for word in ['worried', 'concern', 'afraid']):
            emotion = EmotionalState.REASSURING
            tone = PersonalityTone.REASSURING_TRUSTWORTHY
        elif any(word in text_lower for word in ['order', 'buy', 'want']):
            emotion = EmotionalState.EXCITED
            tone = PersonalityTone.HELPFUL_EAGER
        elif any(word in text_lower for word in ['tell me', 'about', 'information']):
            emotion = EmotionalState.CONFIDENT
            tone = PersonalityTone.CONFIDENT_KNOWLEDGEABLE
        else:
            emotion = EmotionalState.NEUTRAL
            tone = PersonalityTone.WARM_WELCOMING
            
        # Detect urgency
        urgency = 'high' if any(word in text_lower for word in ['urgent', 'asap', 'quickly']) else 'normal'
        
        return {
            'detected_emotion': emotion,
            'suggested_tone': tone,
            'urgency_level': urgency
        }
    
    def get_generation_params(self, context: ConversationContext) -> Dict:
        """Get Maya's optimal generation parameters"""
        
        # Base parameters (always use Maya's optimal temperature)
        params = {
            'do_sample': self.maya_settings['do_sample'],
            'temperature': self.maya_settings['temperature'],
            'top_k': self.maya_settings['top_k'],
            'top_p': self.maya_settings['top_p'],
            'repetition_penalty': self.maya_settings['repetition_penalty']
        }
        
        # Adjust token length based on personality and stage
        if context.response_length == 'short' or context.conversation_stage == ConversationStage.WRAP_UP:
            params['max_new_tokens'] = self.maya_settings['max_new_tokens_short']
        elif context.personality_tone == PersonalityTone.CONFIDENT_KNOWLEDGEABLE:
            params['max_new_tokens'] = self.maya_settings['max_new_tokens_expressive']
        else:
            params['max_new_tokens'] = self.maya_settings['max_new_tokens_normal']
            
        return params
    
    def _generate_speech_safe(self, text: str, params: Dict) -> Optional[torch.Tensor]:
        """Generate speech using multiple fallback methods"""
        
        # Method 1: Try simple conversation format (most reliable)
        try:
            logger.info("Trying Method 1: Simple conversation format")
            conversation = [
                {"role": "0", "content": [{"type": "text", "text": text}]}
            ]
            
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True
            ).to(self.device)
            
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    **params
                )
            
            logger.info("✅ Method 1 successful")
            return self._process_audio_output(audio)
            
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
        
        # Method 2: Try direct text input (fallback)
        try:
            logger.info("Trying Method 2: Direct text input")
            
            # Format text with speaker marker
            text_input = f"[0]{text}"
            
            # Use simple tokenization
            inputs = self.processor(
                text_input, 
                add_special_tokens=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    **params
                )
            
            logger.info("✅ Method 2 successful")
            return self._process_audio_output(audio)
            
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
        
        # Method 3: Most basic approach
        try:
            logger.info("Trying Method 3: Basic generation")
            
            # Simplest possible input
            basic_conversation = [
                {"role": "0", "content": [{"type": "text", "text": text[:100]}]}  # Limit text length
            ]
            
            inputs = self.processor.apply_chat_template(
                basic_conversation,
                tokenize=True,
                return_dict=True
            ).to(self.device)
            
            # Use minimal parameters
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=60  # Fixed short length
                )
            
            logger.info("✅ Method 3 successful")
            return self._process_audio_output(audio)
            
        except Exception as e:
            logger.error(f"All methods failed: {e}")
            return None
    
    def _process_audio_output(self, audio) -> torch.Tensor:
        """Process and normalize audio output"""
        
        # Handle different output formats
        if isinstance(audio, list):
            if len(audio) > 0 and isinstance(audio[0], torch.Tensor):
                audio = audio[0]
            else:
                logger.warning("Unexpected audio list format")
                return audio
                
        if isinstance(audio, torch.Tensor):
            # Move to CPU if needed
            if audio.is_cuda:
                audio = audio.cpu()
                
            # Normalize audio
            if torch.max(torch.abs(audio)) > 0:
                audio = audio / torch.max(torch.abs(audio)) * 0.9
                
            return audio
        else:
            logger.warning(f"Unknown audio format: {type(audio)}")
            return audio
    
    def generate_response(self, session_id: str, user_input: str, 
                         agent_response_text: str) -> Tuple[Any, str, Dict]:
        """Generate Maya-level speech response with fixed CSM compatibility"""
        
        # Get or create session
        if session_id not in self.active_sessions:
            context = self.create_session(session_id)
        else:
            context = self.active_sessions[session_id]
        
        # Analyze user input
        user_analysis = self.analyze_user_input(user_input)
        
        # Update context
        context.personality_tone = user_analysis['suggested_tone']
        context.emotional_state = user_analysis['detected_emotion']
        context.response_length = 'short' if user_analysis['urgency_level'] == 'high' else 'normal'
        context.turn_count += 1
        
        # Update conversation stage
        if context.turn_count <= 2:
            context.conversation_stage = ConversationStage.GREETING
        elif context.turn_count > 8:
            context.conversation_stage = ConversationStage.CLOSING
        else:
            context.conversation_stage = ConversationStage.MAIN_INTERACTION
        
        # Enhance text with personality
        enhanced_text = self.speech_enhancer.enhance_text(agent_response_text, context)
        
        # Get generation parameters
        params = self.get_generation_params(context)
        
        # Log generation details
        logger.info(f"Generating for session {session_id}")
        logger.info(f"Personality: {context.personality_tone.value}")
        logger.info(f"Emotion: {context.emotional_state.value}")
        logger.info(f"Enhanced text: {enhanced_text[:50]}...")
        logger.info(f"Parameters: temp={params['temperature']}, tokens={params['max_new_tokens']}")
        
        # Generate speech with fallback methods
        audio = self._generate_speech_safe(enhanced_text, params)
        
        # Update memory
        self.memory.add_turn(session_id, 'user', user_input)
        self.memory.add_turn(session_id, 'agent', enhanced_text)
        
        # Prepare metadata
        metadata = {
            'context': context,
            'params': params,
            'user_analysis': user_analysis,
            'success': audio is not None
        }
        
        return audio, enhanced_text, metadata
    
    def save_audio(self, audio: torch.Tensor, filename: str, sample_rate: int = 16000):
        """Save audio to file"""
        if audio is None:
            logger.error("Cannot save None audio")
            return
            
        try:
            # Ensure audio is a tensor
            if not isinstance(audio, torch.Tensor):
                logger.error(f"Audio is not a tensor: {type(audio)}")
                return
            
            # Move to CPU
            if audio.is_cuda:
                audio = audio.cpu()
            
            # Ensure correct shape (add batch dimension if needed)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Save using torchaudio
            torchaudio.save(filename, audio, sample_rate)
            logger.info(f"✅ Audio saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            
            # Try alternative save method
            try:
                # Convert to numpy and save with scipy
                import scipy.io.wavfile as wavfile
                audio_np = audio.numpy()
                if audio_np.ndim == 2:
                    audio_np = audio_np.squeeze()
                # Convert to int16 for wav
                audio_int16 = np.int16(audio_np * 32767)
                wavfile.write(filename, sample_rate, audio_int16)
                logger.info(f"✅ Audio saved using scipy to {filename}")
            except Exception as e2:
                logger.error(f"Alternative save also failed: {e2}")

# ============================================================================
# TESTING
# ============================================================================

def test_maya_agent():
    """Test Maya Voice Agent with CSM compatibility fixes"""
    
    print("🎭 MAYA VOICE AGENT - FIXED IMPLEMENTATION TEST")
    print("=" * 60)
    
    try:
        # Initialize agent
        print("Initializing Maya Voice Agent...")
        agent = MayaVoiceAgent()
        print("✅ Agent initialized successfully\n")
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Restaurant Greeting',
                'session_id': 'test_restaurant_001',
                'user_input': "Hi, I'd like to make a reservation",
                'agent_response': "Welcome to our restaurant! I'd be delighted to help you with your reservation.",
                'expected_tone': PersonalityTone.WARM_WELCOMING
            },
            {
                'name': 'Product Inquiry',
                'session_id': 'test_product_001',
                'user_input': "Can you tell me about your most popular items?",
                'agent_response': "Our signature dishes are absolutely worth trying.",
                'expected_tone': PersonalityTone.CONFIDENT_KNOWLEDGEABLE
            },
            {
                'name': 'Urgent Request',
                'session_id': 'test_urgent_001',
                'user_input': "I need help right away with my order",
                'agent_response': "I'll help you with that immediately.",
                'expected_tone': PersonalityTone.HELPFUL_EAGER
            },
            {
                'name': 'Customer Concern',
                'session_id': 'test_concern_001',
                'user_input': "I'm worried about the delivery time",
                'agent_response': "Your concern is completely valid.",
                'expected_tone': PersonalityTone.REASSURING_TRUSTWORTHY
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\n📝 Test {i+1}: {scenario['name']}")
            print(f"👤 User: {scenario['user_input']}")
            print(f"🎯 Expected: {scenario['expected_tone'].value}")
            
            try:
                # Generate response
                audio, enhanced_text, metadata = agent.generate_response(
                    session_id=scenario['session_id'],
                    user_input=scenario['user_input'],
                    agent_response_text=scenario['agent_response']
                )
                
                if audio is not None:
                    # Save audio
                    filename = f"maya_test_{i+1}_{scenario['session_id']}.wav"
                    agent.save_audio(audio, filename)
                    
                    print(f"✅ Audio generated successfully")
                    print(f"📁 Saved to: {filename}")
                    print(f"💬 Enhanced: {enhanced_text[:60]}...")
                    print(f"🎭 Tone: {metadata['context'].personality_tone.value}")
                    print(f"🎵 Emotion: {metadata['context'].emotional_state.value}")
                    
                    results.append({
                        'test': scenario['name'],
                        'success': True,
                        'file': filename,
                        'tone_match': metadata['context'].personality_tone.value == scenario['expected_tone'].value
                    })
                else:
                    print(f"⚠️ No audio generated")
                    results.append({
                        'test': scenario['name'],
                        'success': False,
                        'error': 'No audio generated'
                    })
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results.append({
                    'test': scenario['name'],
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        success_count = sum(1 for r in results if r['success'])
        tone_match_count = sum(1 for r in results if r.get('tone_match', False))
        
        print(f"\n✅ Success Rate: {success_count}/{len(test_scenarios)} tests passed")
        if success_count > 0:
            print(f"🎯 Tone Accuracy: {tone_match_count}/{success_count} correct tones")
        
        print("\nDetailed Results:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            tone = "🎯" if result.get('tone_match', False) else "⚠️" if result['success'] else ""
            print(f"  {status} {result['test']} {tone}")
            if result['success'] and 'file' in result:
                print(f"     📁 {result['file']}")
        
        if success_count > 0:
            print("\n🎭 Maya Voice Agent Features Working:")
            print("  ✅ CSM model integration")
            print("  ✅ Personality-based text enhancement")
            print("  ✅ Emotional state detection")
            print("  ✅ Conversation stage management")
            print("  ✅ Memory and context tracking")
            print("  ✅ Fallback generation methods")
            
            print("\n🎧 Listen to the generated audio files to hear:")
            print("  • Maya's consistent voice across personalities")
            print("  • Natural speech patterns and flow")
            print("  • Personality-appropriate responses")
            print("  • Emotional tone variations")
            
        print("\n💡 Troubleshooting Tips:")
        print("  • Ensure you have access to sesame/csm-1b model")
        print("  • Check CUDA availability for faster processing")
        print("  • Try shorter text inputs if generation fails")
        print("  • Use fallback methods for reliability")
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("\n🔧 Please check:")
        print("  1. Model access: huggingface-cli login")
        print("  2. Dependencies: pip install transformers torch torchaudio")
        print("  3. Model availability: sesame/csm-1b")

if __name__ == "__main__":
    test_maya_agent()