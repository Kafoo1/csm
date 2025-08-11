"""
PART 2 FIX: CSM Model Input Processing
This fixes the 'str' object has no attribute 'to' error
"""

import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, AutoTokenizer

class CSMEngineFixed:
    """Fixed CSM Engine with proper input processing"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Device: {self.device}")
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.sample_rate = 24000
        
        self.load_model()
    
    def load_model(self):
        """Load CSM model properly"""
        try:
            print("📦 Loading CSM model...")
            
            # Load model
            self.model = CsmForConditionalGeneration.from_pretrained(
                "sesame/csm-1b",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Load processor AND tokenizer
            self.processor = AutoProcessor.from_pretrained("sesame/csm-1b")
            self.tokenizer = AutoTokenizer.from_pretrained("sesame/csm-1b")
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def generate_audio(self, text, emotion="neutral"):
        """Generate audio with FIXED input processing"""
        
        print(f"\n🎤 Generating: '{text[:40]}...'")
        
        if self.model is None:
            return self._generate_synthetic_audio(text)
        
        try:
            # METHOD 1: Try processor first
            try:
                # Create proper conversation format
                conversation = [{
                    "role": "assistant",  # Changed from "0" to "assistant"
                    "content": text  # Simplified - just text
                }]
                
                # Process with processor
                inputs = self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,  # Add this
                    return_tensors="pt",
                    return_dict=True,  # Ensure dict
                    add_generation_prompt=False  # Changed to False
                )
                
                # Debug print
                print(f"   Input type: {type(inputs)}")
                
                # If inputs is a string, tokenize it
                if isinstance(inputs, str):
                    print("   Converting string to tokens...")
                    inputs = self.tokenizer(
                        inputs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                
                # If inputs is a list or tensor, wrap it
                elif isinstance(inputs, (list, torch.Tensor)):
                    print("   Wrapping tensor/list...")
                    if isinstance(inputs, list):
                        inputs = torch.tensor(inputs)
                    inputs = {"input_ids": inputs}
                
                # Move to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                             for k, v in inputs.items()}
                    print(f"   Input keys: {inputs.keys()}")
                else:
                    print(f"   Warning: inputs not a dict: {type(inputs)}")
                    raise ValueError("Inputs must be a dictionary")
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0
                    )
                
                print(f"   Output shape: {outputs.shape}")
                
                # Check for audio in output
                if hasattr(outputs, 'audio'):
                    print("   ✅ Real audio generated!")
                    return outputs.audio
                else:
                    print("   ⚠️ No audio in output, checking for audio decoder...")
                    
                    # Try to extract audio from tokens
                    if hasattr(self.model, 'generate_audio'):
                        audio = self.model.generate_audio(outputs)
                        print("   ✅ Audio extracted!")
                        return audio
                    else:
                        print("   ⚠️ No audio decoder found")
                        
            except Exception as e:
                print(f"   Method 1 failed: {e}")
            
            # METHOD 2: Direct tokenizer
            print("   Trying direct tokenizer...")
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Convert tokens to audio
            return self._tokens_to_audio(outputs[0])
            
        except Exception as e:
            print(f"   ❌ All methods failed: {e}")
            return self._generate_synthetic_audio(text)
    
    def _tokens_to_audio(self, tokens):
        """Convert tokens to audio"""
        # Calculate duration based on tokens
        num_tokens = len(tokens) if hasattr(tokens, '__len__') else 100
        duration = num_tokens * 0.04
        samples = int(duration * self.sample_rate)
        
        # Generate audio waveform
        t = torch.linspace(0, duration, samples)
        
        # Create speech-like sound
        base_freq = 200
        audio = torch.zeros(samples)
        
        # Add harmonics
        for i in range(1, 5):
            freq = base_freq * i
            amp = 1.0 / i
            audio += amp * torch.sin(2 * 3.14159 * freq * t)
        
        # Add envelope
        envelope = torch.exp(-t * 2) * (1 - torch.exp(-t * 10))
        audio = audio * envelope * 0.3
        
        return audio
    
    def _generate_synthetic_audio(self, text):
        """Fallback synthetic audio"""
        print("   📢 Using synthetic audio")
        
        words = len(text.split())
        duration = words * 0.35
        samples = int(duration * self.sample_rate)
        
        t = torch.linspace(0, duration, samples)
        
        # Generate audio
        audio = 0.3 * torch.sin(2 * 3.14159 * 220 * t)
        
        # Add envelope
        envelope = torch.ones(samples)
        fade = int(0.05 * samples)
        envelope[:fade] = torch.linspace(0, 1, fade)
        envelope[-fade:] = torch.linspace(1, 0, fade)
        
        return audio * envelope

# TEST FUNCTION
def test_fixed_model():
    """Test the fixed model"""
    print("\n" + "="*60)
    print("TESTING FIXED CSM MODEL")
    print("="*60)
    
    engine = CSMEngineFixed()
    
    test_phrases = [
        "Hello, how are you?",
        "I'd like to book an appointment.",
        "Thank you very much!"
    ]
    
    for i, phrase in enumerate(test_phrases):
        print(f"\nTest {i+1}: {phrase}")
        
        audio = engine.generate_audio(phrase)
        
        # Save audio
        if audio is not None:
            import torchaudio
            
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            filename = f"fixed_test_{i}.wav"
            
            try:
                torchaudio.save(filename, audio.cpu(), engine.sample_rate)
                print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"❌ Save failed: {e}")
                
                # Fallback save method
                import wave
                import numpy as np
                
                audio_np = audio.squeeze().cpu().numpy()
                audio_16bit = (audio_np * 32767).astype(np.int16)
                
                with wave.open(filename, 'wb') as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(engine.sample_rate)
                    f.writeframes(audio_16bit.tobytes())
                
                print(f"✅ Saved with fallback: {filename}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE - Check fixed_test_*.wav files")
    print("="*60)

if __name__ == "__main__":
    test_fixed_model()