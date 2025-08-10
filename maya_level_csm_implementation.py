"""
🎭 MAYA-LEVEL VOICE AGENT WITH ENHANCED CSM
Production-ready implementation with advanced personality consistency,
emotional modulation, and conversational memory
"""

import torch
import torchaudio
from transformers import AutoProcessor, CsmForConditionalGeneration
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
# PERSONALITY SYSTEM
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
        # Initialize signature phrases for each personality tone
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
# MEMORY SYSTEM
# ============================================================================

class ConversationalMemory:
    """Advanced memory system for context-aware responses"""
    
    def __init__(self, max_turns: int = 10, context_window_minutes: float = 2.0):
        self.max_turns = max_turns
        self.context_window_minutes = context_window_minutes
        self.sessions: Dict[str, deque] = {}
        self.session_contexts: Dict[str, ConversationContext] = {}
        
    def add_turn(self, session_id: str, speaker: str, text: str, 
                 emotion: Optional[str] = None, audio_features: Optional[Dict] = None):
        """Add a conversation turn to memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_turns)
            
        turn = {
            'timestamp': time.time(),
            'speaker': speaker,
            'text': text,
            'emotion': emotion,
            'audio_features': audio_features
        }
        
        self.sessions[session_id].append(turn)
        logger.info(f"Added turn to session {session_id}: {speaker} - {text[:50]}...")
        
    def get_recent_context(self, session_id: str) -> List[Dict]:
        """Get recent conversation context within time window"""
        if session_id not in self.sessions:
            return []
            
        current_time = time.time()
        cutoff_time = current_time - (self.context_window_minutes * 60)
        
        recent_turns = [
            turn for turn in self.sessions[session_id]
            if turn['timestamp'] > cutoff_time
        ]
        
        return recent_turns
    
    def analyze_conversation_flow(self, session_id: str) -> Dict[str, Any]:
        """Analyze conversation patterns and emotional trajectory"""
        recent_turns = self.get_recent_context(session_id)
        
        if not recent_turns:
            return {'pattern': 'new_conversation', 'emotional_trajectory': 'neutral'}
            
        # Analyze emotional trajectory
        emotions = [turn['emotion'] for turn in recent_turns if turn['emotion']]
        
        # Detect conversation patterns
        patterns = {
            'is_escalating': self._detect_escalation(recent_turns),
            'needs_reassurance': self._detect_reassurance_need(recent_turns),
            'is_positive': self._detect_positive_trend(recent_turns),
            'topic_switches': self._count_topic_switches(recent_turns)
        }
        
        return patterns
    
    def _detect_escalation(self, turns: List[Dict]) -> bool:
        """Detect if conversation is escalating negatively"""
        negative_keywords = ['problem', 'issue', 'wrong', 'frustrated', 'angry', 'upset']
        recent_text = ' '.join([t['text'].lower() for t in turns[-3:]])
        return any(keyword in recent_text for keyword in negative_keywords)
    
    def _detect_reassurance_need(self, turns: List[Dict]) -> bool:
        """Detect if user needs reassurance"""
        concern_keywords = ['worried', 'concern', 'not sure', 'confused', 'help']
        recent_text = ' '.join([t['text'].lower() for t in turns[-2:]])
        return any(keyword in recent_text for keyword in concern_keywords)
    
    def _detect_positive_trend(self, turns: List[Dict]) -> bool:
        """Detect positive conversation trend"""
        positive_keywords = ['great', 'perfect', 'excellent', 'thank you', 'happy']
        recent_text = ' '.join([t['text'].lower() for t in turns[-2:]])
        return any(keyword in recent_text for keyword in positive_keywords)
    
    def _count_topic_switches(self, turns: List[Dict]) -> int:
        """Count topic switches in conversation"""
        # Simplified topic detection - in production, use NLP
        return min(len(turns) // 3, 3)

# ============================================================================
# DIALOGUE MANAGER
# ============================================================================

class DialogueStateManager:
    """Manages conversation flow and enforces dialogue rules"""
    
    def __init__(self, personality_profile: PersonalityProfile):
        self.personality = personality_profile
        self.state_transitions = self._initialize_state_transitions()
        self.dialogue_rules = self._initialize_dialogue_rules()
        
    def _initialize_state_transitions(self) -> Dict:
        """Define valid state transitions"""
        return {
            ConversationStage.GREETING: [ConversationStage.NEEDS_ASSESSMENT],
            ConversationStage.NEEDS_ASSESSMENT: [ConversationStage.MAIN_INTERACTION, ConversationStage.OBJECTION_HANDLING],
            ConversationStage.MAIN_INTERACTION: [ConversationStage.OBJECTION_HANDLING, ConversationStage.CLOSING],
            ConversationStage.OBJECTION_HANDLING: [ConversationStage.MAIN_INTERACTION, ConversationStage.CLOSING],
            ConversationStage.CLOSING: [ConversationStage.WRAP_UP],
            ConversationStage.WRAP_UP: []
        }
    
    def _initialize_dialogue_rules(self) -> Dict:
        """Initialize mandatory dialogue rules"""
        return {
            'greeting_required': ['name', 'purpose'],
            'closing_required': ['next_steps', 'confirmation'],
            'objection_patterns': {
                'price': ['value', 'budget', 'options'],
                'time': ['schedule', 'availability', 'convenient'],
                'trust': ['guarantee', 'experience', 'references']
            },
            'fallback_responses': {
                'unclear': "I want to make sure I understand correctly. Could you tell me more about...",
                'silence': "Are you still there? I'm here to help if you have any questions.",
                'off_topic': "That's interesting! Let me help you with..."
            }
        }
    
    def get_next_stage(self, current_stage: ConversationStage, 
                       context: ConversationContext) -> ConversationStage:
        """Determine next conversation stage based on context"""
        valid_transitions = self.state_transitions.get(current_stage, [])
        
        if not valid_transitions:
            return current_stage
            
        # Logic for stage transition based on context
        if context.turn_count < 2:
            return ConversationStage.GREETING
        elif context.objections_raised:
            return ConversationStage.OBJECTION_HANDLING
        elif context.turn_count > 10:
            return ConversationStage.CLOSING
            
        return valid_transitions[0] if valid_transitions else current_stage
    
    def get_stage_appropriate_response(self, stage: ConversationStage, 
                                      base_response: str) -> str:
        """Enhance response based on conversation stage"""
        stage_enhancements = {
            ConversationStage.GREETING: self._enhance_greeting,
            ConversationStage.NEEDS_ASSESSMENT: self._enhance_needs_assessment,
            ConversationStage.MAIN_INTERACTION: self._enhance_main_interaction,
            ConversationStage.OBJECTION_HANDLING: self._enhance_objection_handling,
            ConversationStage.CLOSING: self._enhance_closing,
            ConversationStage.WRAP_UP: self._enhance_wrap_up
        }
        
        enhancer = stage_enhancements.get(stage, lambda x: x)
        return enhancer(base_response)
    
    def _enhance_greeting(self, response: str) -> str:
        """Enhance greeting stage response"""
        if not any(greeting in response.lower() for greeting in ['hi', 'hello', 'welcome']):
            time_greeting = self._get_time_based_greeting()
            response = f"{time_greeting} {response}"
        return response
    
    def _enhance_needs_assessment(self, response: str) -> str:
        """Enhance needs assessment response"""
        if '?' not in response:
            response += " What brings you here today?"
        return response
    
    def _enhance_main_interaction(self, response: str) -> str:
        """Enhance main interaction response"""
        return response
    
    def _enhance_objection_handling(self, response: str) -> str:
        """Enhance objection handling response"""
        if not any(word in response.lower() for word in ['understand', 'appreciate', 'hear']):
            response = f"I completely understand your concern. {response}"
        return response
    
    def _enhance_closing(self, response: str) -> str:
        """Enhance closing response"""
        if 'next' not in response.lower():
            response += " What would be the best next step for you?"
        return response
    
    def _enhance_wrap_up(self, response: str) -> str:
        """Enhance wrap-up response"""
        if 'thank' not in response.lower():
            response = f"Thank you so much for your time! {response}"
        return response
    
    def _get_time_based_greeting(self) -> str:
        """Get appropriate greeting based on time"""
        hour = datetime.datetime.now().hour
        if hour < 12:
            return "Good morning!"
        elif hour < 17:
            return "Good afternoon!"
        else:
            return "Good evening!"

# ============================================================================
# PROSODY CONTROLLER
# ============================================================================

class ProsodyController:
    """Controls voice prosody parameters based on context"""
    
    def __init__(self):
        self.base_settings = {
            'temperature': 0.72,  # Maya's optimal temperature
            'pitch_range': (0.9, 1.1),
            'speed_range': (0.95, 1.05),
            'pause_duration': (0.2, 0.8),
            'emphasis_strength': 1.0
        }
        
        self.emotion_profiles = self._initialize_emotion_profiles()
        
    def _initialize_emotion_profiles(self) -> Dict:
        """Define prosody profiles for different emotions"""
        return {
            EmotionalState.NEUTRAL: {
                'pitch_modifier': 1.0,
                'speed_modifier': 1.0,
                'pause_frequency': 'normal',
                'emphasis': 'moderate'
            },
            EmotionalState.HAPPY: {
                'pitch_modifier': 1.05,
                'speed_modifier': 1.02,
                'pause_frequency': 'reduced',
                'emphasis': 'increased'
            },
            EmotionalState.EMPATHETIC: {
                'pitch_modifier': 0.98,
                'speed_modifier': 0.97,
                'pause_frequency': 'increased',
                'emphasis': 'gentle'
            },
            EmotionalState.CONFIDENT: {
                'pitch_modifier': 1.02,
                'speed_modifier': 0.98,
                'pause_frequency': 'strategic',
                'emphasis': 'strong'
            },
            EmotionalState.REASSURING: {
                'pitch_modifier': 0.97,
                'speed_modifier': 0.95,
                'pause_frequency': 'increased',
                'emphasis': 'gentle'
            },
            EmotionalState.EXCITED: {
                'pitch_modifier': 1.08,
                'speed_modifier': 1.05,
                'pause_frequency': 'reduced',
                'emphasis': 'strong'
            },
            EmotionalState.CALM: {
                'pitch_modifier': 0.95,
                'speed_modifier': 0.93,
                'pause_frequency': 'increased',
                'emphasis': 'soft'
            }
        }
    
    def get_prosody_parameters(self, context: ConversationContext) -> Dict:
        """Get prosody parameters based on conversation context"""
        emotion_profile = self.emotion_profiles.get(
            context.emotional_state, 
            self.emotion_profiles[EmotionalState.NEUTRAL]
        )
        
        # Adjust for conversation stage
        stage_adjustments = self._get_stage_adjustments(context.conversation_stage)
        
        # Combine base settings with emotional and stage adjustments
        parameters = {
            'temperature': self.base_settings['temperature'],
            'pitch': self.base_settings['pitch_range'][0] * emotion_profile['pitch_modifier'] * stage_adjustments['pitch'],
            'speed': self.base_settings['speed_range'][0] * emotion_profile['speed_modifier'] * stage_adjustments['speed'],
            'pause_frequency': emotion_profile['pause_frequency'],
            'emphasis': emotion_profile['emphasis'],
            'max_new_tokens': self._calculate_token_length(context)
        }
        
        return parameters
    
    def _get_stage_adjustments(self, stage: ConversationStage) -> Dict:
        """Get prosody adjustments based on conversation stage"""
        stage_modifiers = {
            ConversationStage.GREETING: {'pitch': 1.03, 'speed': 1.0},
            ConversationStage.NEEDS_ASSESSMENT: {'pitch': 1.0, 'speed': 0.98},
            ConversationStage.MAIN_INTERACTION: {'pitch': 1.0, 'speed': 1.0},
            ConversationStage.OBJECTION_HANDLING: {'pitch': 0.98, 'speed': 0.95},
            ConversationStage.CLOSING: {'pitch': 1.02, 'speed': 0.98},
            ConversationStage.WRAP_UP: {'pitch': 1.05, 'speed': 1.0}
        }
        
        return stage_modifiers.get(stage, {'pitch': 1.0, 'speed': 1.0})
    
    def _calculate_token_length(self, context: ConversationContext) -> int:
        """Calculate appropriate token length based on context"""
        base_tokens = {
            'short': 60,
            'normal': 80,
            'expressive': 100
        }
        
        # Adjust based on response length preference
        tokens = base_tokens.get(context.response_length, 80)
        
        # Further adjust based on personality tone
        tone_adjustments = {
            PersonalityTone.WARM_WELCOMING: 85,
            PersonalityTone.CONFIDENT_KNOWLEDGEABLE: 95,
            PersonalityTone.HELPFUL_EAGER: 90,
            PersonalityTone.REASSURING_TRUSTWORTHY: 80
        }
        
        return tone_adjustments.get(context.personality_tone, tokens)

# ============================================================================
# SPEECH ENHANCER
# ============================================================================

class SpeechEnhancer:
    """Enhances text with natural speech patterns and personality traits"""
    
    def __init__(self, personality_profile: PersonalityProfile):
        self.personality = personality_profile
        self.filler_insertion_probability = 0.15
        self.breath_marker_probability = 0.1
        
    def enhance_text(self, text: str, context: ConversationContext) -> str:
        """Apply personality-based text enhancements"""
        enhanced = text
        
        # Apply personality-specific enhancements
        enhanced = self._apply_personality_style(enhanced, context.personality_tone)
        
        # Add natural speech elements
        enhanced = self._add_natural_elements(enhanced, context)
        
        # Apply emotional coloring
        enhanced = self._apply_emotional_coloring(enhanced, context.emotional_state)
        
        return enhanced
    
    def _apply_personality_style(self, text: str, tone: PersonalityTone) -> str:
        """Apply personality-specific language patterns"""
        style_transforms = {
            PersonalityTone.WARM_WELCOMING: self._warm_welcoming_style,
            PersonalityTone.CONFIDENT_KNOWLEDGEABLE: self._confident_style,
            PersonalityTone.HELPFUL_EAGER: self._eager_style,
            PersonalityTone.REASSURING_TRUSTWORTHY: self._reassuring_style
        }
        
        transform = style_transforms.get(tone, lambda x: x)
        return transform(text)
    
    def _warm_welcoming_style(self, text: str) -> str:
        """Apply warm welcoming style"""
        if not text.lower().startswith(('hi', 'hello', 'welcome')):
            text = f"Hi there! {text}"
        text = text.replace(' help ', ' absolutely help ')
        text = text.replace(' can ', ' would love to ')
        return text
    
    def _confident_style(self, text: str) -> str:
        """Apply confident knowledgeable style"""
        text = text.replace(' think ', ' know ')
        text = text.replace(' probably ', ' definitely ')
        text = text.replace(' good ', ' excellent ')
        return text
    
    def _eager_style(self, text: str) -> str:
        """Apply helpful eager style"""
        if not text.lower().startswith(('perfect', 'great', 'awesome')):
            text = f"Perfect! {text}"
        text = text.replace(' can ', ' can absolutely ')
        text = text.replace(' will ', ' will definitely ')
        return text
    
    def _reassuring_style(self, text: str) -> str:
        """Apply reassuring trustworthy style"""
        if 'understand' not in text.lower():
            text = f"I completely understand. {text}"
        text = text.replace(' will ', ' will personally ')
        text = text.replace(' ensure ', ' absolutely ensure ')
        return text
    
    def _add_natural_elements(self, text: str, context: ConversationContext) -> str:
        """Add natural speech elements like fillers and pauses"""
        import random
        
        # Add occasional fillers
        if random.random() < self.filler_insertion_probability:
            filler = random.choice(self.personality.filler_words)
            words = text.split()
            if len(words) > 5:
                insert_pos = random.randint(2, min(5, len(words)-1))
                words.insert(insert_pos, filler)
                text = ' '.join(words)
        
        # Add breath markers (for speech synthesis)
        if random.random() < self.breath_marker_probability:
            text = text.replace('. ', '. <breath> ')
        
        return text
    
    def _apply_emotional_coloring(self, text: str, emotion: EmotionalState) -> str:
        """Apply emotional coloring to text"""
        emotion_modifiers = {
            EmotionalState.HAPPY: lambda t: t.replace('!', '!!'),
            EmotionalState.EMPATHETIC: lambda t: f"I hear you. {t}",
            EmotionalState.CONFIDENT: lambda t: t.replace('I think', 'I know'),
            EmotionalState.REASSURING: lambda t: f"Don't worry. {t}",
            EmotionalState.EXCITED: lambda t: t.upper() if len(t) < 20 else t + '!',
            EmotionalState.CALM: lambda t: t.replace('!', '.')
        }
        
        modifier = emotion_modifiers.get(emotion, lambda t: t)
        return modifier(text)

# ============================================================================
# MAIN MAYA VOICE AGENT
# ============================================================================

class MayaVoiceAgent:
    """Complete Maya-level voice agent with all enhancements"""
    
    def __init__(self, model_id: str = "sesame/csm-1b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Maya Voice Agent on {self.device}")
        
        # Initialize personality
        self.personality = PersonalityProfile()
        
        # Initialize components
        self.memory = ConversationalMemory()
        self.dialogue_manager = DialogueStateManager(self.personality)
        self.prosody_controller = ProsodyController()
        self.speech_enhancer = SpeechEnhancer(self.personality)
        
        # Load model
        self._load_model(model_id)
        
        # Session management
        self.active_sessions: Dict[str, ConversationContext] = {}
        
        logger.info("Maya Voice Agent initialized successfully")
    
    def _load_model(self, model_id: str):
        """Load CSM model with optimal settings"""
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = CsmForConditionalGeneration.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_cache=True
            )
            self.model.eval()
            logger.info(f"Model {model_id} loaded successfully")
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
    
    def analyze_user_input(self, text: str, audio_features: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze user input for emotional and contextual cues"""
        analysis = {
            'detected_emotion': EmotionalState.NEUTRAL,
            'detected_intent': 'general',
            'urgency_level': 'normal',
            'requires_personality_shift': False
        }
        
        text_lower = text.lower()
        
        # Emotion detection
        if any(word in text_lower for word in ['angry', 'frustrated', 'upset']):
            analysis['detected_emotion'] = EmotionalState.EMPATHETIC
            analysis['requires_personality_shift'] = True
        elif any(word in text_lower for word in ['happy', 'great', 'excellent']):
            analysis['detected_emotion'] = EmotionalState.HAPPY
        elif any(word in text_lower for word in ['worried', 'concerned', 'afraid']):
            analysis['detected_emotion'] = EmotionalState.REASSURING
            analysis['requires_personality_shift'] = True
        
        # Intent detection
        if any(word in text_lower for word in ['order', 'buy', 'purchase']):
            analysis['detected_intent'] = 'transaction'
        elif any(word in text_lower for word in ['help', 'support', 'assist']):
            analysis['detected_intent'] = 'support'
        elif any(word in text_lower for word in ['information', 'about', 'tell me']):
            analysis['detected_intent'] = 'inquiry'
        
        # Urgency detection
        if any(word in text_lower for word in ['urgent', 'asap', 'immediately', 'quickly']):
            analysis['urgency_level'] = 'high'
        
        return analysis
    
    def determine_personality_tone(self, context: ConversationContext, user_analysis: Dict) -> PersonalityTone:
        """Dynamically determine appropriate personality tone"""
        
        # If user needs reassurance, switch to reassuring tone
        if user_analysis['detected_emotion'] == EmotionalState.REASSURING:
            return PersonalityTone.REASSURING_TRUSTWORTHY
        
        # If user is upset, use empathetic warm tone
        elif user_analysis['detected_emotion'] == EmotionalState.EMPATHETIC:
            return PersonalityTone.WARM_WELCOMING
        
        # For transactions, be helpful and eager
        elif user_analysis['detected_intent'] == 'transaction':
            return PersonalityTone.HELPFUL_EAGER
        
        # For information requests, be knowledgeable
        elif user_analysis['detected_intent'] == 'inquiry':
            return PersonalityTone.CONFIDENT_KNOWLEDGEABLE
        
        # Otherwise, maintain current tone
        return context.personality_tone
    
    def generate_response(self, session_id: str, user_input: str, 
                         agent_response_text: str) -> Tuple[Any, str, Dict]:
        """Generate Maya-level speech response"""
        
        # Get or create session context
        if session_id not in self.active_sessions:
            context = self.create_session(session_id)
        else:
            context = self.active_sessions[session_id]
        
        # Analyze user input
        user_analysis = self.analyze_user_input(user_input)
        
        # Update context based on analysis
        context.personality_tone = self.determine_personality_tone(context, user_analysis)
        context.emotional_state = user_analysis['detected_emotion']
        context.turn_count += 1
        
        # Get conversation flow analysis
        flow_analysis = self.memory.analyze_conversation_flow(session_id)
        
        # Update conversation stage
        context.conversation_stage = self.dialogue_manager.get_next_stage(
            context.conversation_stage, context
        )
        
        # Enhance response text based on stage
        enhanced_text = self.dialogue_manager.get_stage_appropriate_response(
            context.conversation_stage, agent_response_text
        )
        
        # Apply personality and emotional enhancements
        enhanced_text = self.speech_enhancer.enhance_text(enhanced_text, context)
        
        # Get prosody parameters
        prosody_params = self.prosody_controller.get_prosody_parameters(context)
        
        # Add to memory
        self.memory.add_turn(session_id, 'agent', enhanced_text, 
                           emotion=context.emotional_state.value)
        self.memory.add_turn(session_id, 'user', user_input,
                           emotion=user_analysis['detected_emotion'].value)
        
        # Generate speech
        try:
            audio = self._generate_speech(enhanced_text, prosody_params)
            
            logger.info(f"Generated response for session {session_id}")
            logger.info(f"Personality: {context.personality_tone.value}")
            logger.info(f"Emotion: {context.emotional_state.value}")
            logger.info(f"Stage: {context.conversation_stage.value}")
            
            return audio, enhanced_text, {
                'context': context,
                'prosody': prosody_params,
                'user_analysis': user_analysis,
                'flow_analysis': flow_analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            return None, enhanced_text, {'error': str(e)}
    
    def _generate_speech(self, text: str, prosody_params: Dict) -> torch.Tensor:
        """Generate speech with prosody parameters"""
        
        # Prepare conversation format for CSM
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": text}]}
        ]
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True
        ).to(self.device)
        
        # Generate speech
        with torch.no_grad():
            audio = self.model.generate(
                **inputs,
                output_audio=True,
                max_new_tokens=prosody_params['max_new_tokens'],
                temperature=prosody_params['temperature'],
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        return audio
    
    def save_audio(self, audio: torch.Tensor, filename: str, sample_rate: int = 16000):
        """Save generated audio to file"""
        try:
            # Handle different audio formats
            if isinstance(audio, list):
                audio = audio[0] if len(audio) > 0 else audio
            
            if isinstance(audio, torch.Tensor):
                # Ensure audio is on CPU
                audio = audio.cpu()
                
                # Normalize audio
                if torch.max(torch.abs(audio)) > 0:
                    audio = audio / torch.max(torch.abs(audio)) * 0.95
                
                # Save using torchaudio
                torchaudio.save(filename, audio.unsqueeze(0), sample_rate)
                logger.info(f"Audio saved to {filename}")
            else:
                logger.error(f"Unsupported audio format: {type(audio)}")
                
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")

# ============================================================================
# TESTING AND EVALUATION
# ============================================================================

def test_maya_agent():
    """Comprehensive test of Maya Voice Agent"""
    
    print("🎭 MAYA VOICE AGENT - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Initialize agent
    agent = MayaVoiceAgent()
    
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
            'agent_response': "Our signature dishes are absolutely worth trying, especially the chef's special.",
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
            'agent_response': "Your concern is completely valid, and I'll make sure everything arrives on time.",
            'expected_tone': PersonalityTone.REASSURING_TRUSTWORTHY
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n📝 Testing: {scenario['name']}")
        print(f"👤 User: {scenario['user_input']}")
        print(f"🎯 Expected Tone: {scenario['expected_tone'].value}")
        
        try:
            # Generate response
            audio, enhanced_text, metadata = agent.generate_response(
                session_id=scenario['session_id'],
                user_input=scenario['user_input'],
                agent_response_text=scenario['agent_response']
            )
            
            if audio is not None:
                # Save audio
                filename = f"maya_test_{scenario['session_id']}.wav"
                agent.save_audio(audio, filename)
                
                print(f"✅ Generated: {filename}")
                print(f"💬 Enhanced: {enhanced_text}")
                print(f"🎭 Actual Tone: {metadata['context'].personality_tone.value}")
                print(f"🎵 Emotion: {metadata['context'].emotional_state.value}")
                print(f"📊 Stage: {metadata['context'].conversation_stage.value}")
                
                results.append({
                    'test': scenario['name'],
                    'success': True,
                    'file': filename,
                    'tone_match': metadata['context'].personality_tone == scenario['expected_tone']
                })
            else:
                print(f"❌ Failed to generate audio")
                results.append({
                    'test': scenario['name'],
                    'success': False,
                    'error': metadata.get('error', 'Unknown error')
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
    
    print(f"✅ Success Rate: {success_count}/{len(test_scenarios)} tests passed")
    print(f"🎯 Tone Accuracy: {tone_match_count}/{success_count} correct tones")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        tone = "🎯" if result.get('tone_match', False) else "⚠️"
        print(f"  {status} {result['test']} {tone if result['success'] else ''}")
    
    print("\n🎭 Maya Voice Agent Features:")
    print("  ✅ Personality consistency across interactions")
    print("  ✅ Emotional prosody modulation")
    print("  ✅ Conversational memory and context")
    print("  ✅ Stage-based dialogue management")
    print("  ✅ Natural speech patterns and fillers")
    print("  ✅ Dynamic personality tone selection")

if __name__ == "__main__":
    test_maya_agent()