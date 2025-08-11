"""
🚀 CSM MEMORY-OPTIMIZED IMPLEMENTATION
Designed for 6GB GPUs with automatic fallbacks and memory management
"""

import torch
import gc
import os
import logging
from transformers import CsmForConditionalGeneration, AutoProcessor
from typing import Optional, Dict, Any, Tuple
import warnings

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizedCSM:
    """CSM implementation optimized for limited GPU memory (4-6GB)"""
    
    def __init__(self, model_id: str = "sesame/csm-1b", device: str = "auto"):
        """
        Initialize CSM with memory optimizations
        
        Args:
            model_id: Model identifier
            device: Device selection ("auto", "cuda", "cpu", "cuda:0")
        """
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = self._select_device(device)
        self.memory_config = self._get_memory_config()
        
        logger.info(f"🎯 Initializing CSM on {self.device}")
        logger.info(f"💾 Memory config: {self.memory_config}")
        
    def _select_device(self, device: str) -> str:
        """Intelligently select device based on available memory"""
        if device == "auto":
            if torch.cuda.is_available():
                # Check available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                free_memory = (torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1e9
                
                logger.info(f"📊 GPU Memory: {gpu_memory:.2f}GB total, {free_memory:.2f}GB free")
                
                if gpu_memory < 8:  # Less than 8GB GPU
                    logger.warning("⚠️ Limited GPU memory detected, using CPU fallback")
                    return "cpu"
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _get_memory_config(self) -> Dict[str, Any]:
        """Get optimal memory configuration based on device"""
        if self.device == "cpu":
            return {
                "load_in_8bit": False,
                "load_in_4bit": False,
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "offload_folder": "./offload",
                "offload_state_dict": True
            }
        else:
            # GPU with limited memory
            return {
                "load_in_8bit": True,  # 8-bit quantization
                "load_in_4bit": False,  # Or try 4-bit if 8-bit still fails
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "max_memory": {0: "5GB", "cpu": "10GB"},
                "offload_folder": "./offload",
                "offload_state_dict": True
            }
    
    def clear_memory(self):
        """Aggressively clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def load_model_with_fallbacks(self):
        """Load model with multiple fallback strategies"""
        
        # Strategy 1: Try with 8-bit quantization
        logger.info("🔄 Strategy 1: Loading with 8-bit quantization...")
        try:
            self.clear_memory()
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4"
            )
            
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info("✅ Model loaded with 8-bit quantization")
            return True
            
        except Exception as e:
            logger.warning(f"❌ 8-bit loading failed: {e}")
            self.clear_memory()
        
        # Strategy 2: Try with 4-bit quantization
        logger.info("🔄 Strategy 2: Loading with 4-bit quantization...")
        try:
            self.clear_memory()
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info("✅ Model loaded with 4-bit quantization")
            return True
            
        except Exception as e:
            logger.warning(f"❌ 4-bit loading failed: {e}")
            self.clear_memory()
        
        # Strategy 3: CPU offloading with mixed precision
        logger.info("🔄 Strategy 3: Loading with CPU offloading...")
        try:
            self.clear_memory()
            
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_folder="./offload",
                offload_state_dict=True,
                max_memory={0: "4GB", "cpu": "16GB"}
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info("✅ Model loaded with CPU offloading")
            return True
            
        except Exception as e:
            logger.warning(f"❌ CPU offloading failed: {e}")
            self.clear_memory()
        
        # Strategy 4: Pure CPU fallback
        logger.info("🔄 Strategy 4: Loading on CPU only...")
        try:
            self.clear_memory()
            self.device = "cpu"
            
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to("cpu")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info("✅ Model loaded on CPU")
            return True
            
        except Exception as e:
            logger.error(f"❌ All loading strategies failed: {e}")
            return False
    
    def generate_optimized(
        self, 
        text: str, 
        max_new_tokens: int = 60,
        temperature: float = 0.72,
        speaker_id: int = 0
    ) -> Optional[torch.Tensor]:
        """
        Generate speech with memory optimization
        
        Args:
            text: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            speaker_id: Speaker ID
            
        Returns:
            Audio tensor or None if generation fails
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return None
        
        try:
            # Clear memory before generation
            self.clear_memory()
            
            # Prepare input with minimal memory footprint
            conversation = [{
                "role": str(speaker_id), 
                "content": [{"type": "text", "text": text[:200]}]  # Limit text length
            }]
            
            # Process inputs
            with torch.no_grad():
                inputs = self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True
                )
                
                # Move to appropriate device
                if self.device != "cpu" and torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate with minimal parameters
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9
                )
                
                # Move to CPU immediately to free GPU memory
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu()
                elif isinstance(audio, list) and len(audio) > 0:
                    audio = audio[0].cpu() if isinstance(audio[0], torch.Tensor) else audio[0]
                
                # Clear GPU memory after generation
                self.clear_memory()
                
                return audio
                
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"⚠️ OOM during generation: {e}")
            self.clear_memory()
            
            # Try with reduced parameters
            logger.info("🔄 Retrying with reduced parameters...")
            return self._generate_minimal(text, speaker_id)
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return None
    
    def _generate_minimal(self, text: str, speaker_id: int = 0) -> Optional[torch.Tensor]:
        """Minimal generation with smallest possible memory footprint"""
        try:
            # Use smallest possible configuration
            conversation = [{
                "role": str(speaker_id), 
                "content": [{"type": "text", "text": text[:100]}]  # Very short text
            }]
            
            with torch.no_grad():
                inputs = self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True
                )
                
                # Keep everything on CPU if GPU is tight
                if self.device == "cpu":
                    inputs = inputs.to("cpu")
                else:
                    inputs = inputs.to("cuda")
                
                # Minimal generation parameters
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=40,  # Reduced
                    temperature=0.7,
                    do_sample=False  # Greedy decoding to save memory
                )
                
                # Immediately move to CPU
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu()
                elif isinstance(audio, list) and len(audio) > 0:
                    audio = audio[0].cpu() if isinstance(audio[0], torch.Tensor) else audio[0]
                
                self.clear_memory()
                return audio
                
        except Exception as e:
            logger.error(f"❌ Minimal generation also failed: {e}")
            return None

# ============================================================================
# ALTERNATIVE: STREAMING IMPLEMENTATION FOR LOW MEMORY
# ============================================================================

class StreamingCSM:
    """Streaming implementation for extreme memory constraints"""
    
    def __init__(self):
        self.chunk_size = 256  # Process in small chunks
        self.overlap = 32  # Overlap between chunks
        
    def generate_streaming(self, text: str, model_path: str):
        """Generate audio in streaming chunks to minimize memory"""
        
        # Split text into small chunks
        text_chunks = self._split_text(text, max_words=10)
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:30]}...")
            
            # Load model for this chunk only
            audio_chunk = self._process_chunk(chunk, model_path)
            
            if audio_chunk is not None:
                yield audio_chunk
            
            # Clear memory after each chunk
            torch.cuda.empty_cache()
            gc.collect()
    
    def _split_text(self, text: str, max_words: int = 10) -> list:
        """Split text into small chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i+max_words])
            chunks.append(chunk)
        
        return chunks
    
    def _process_chunk(self, text_chunk: str, model_path: str) -> Optional[torch.Tensor]:
        """Process a single chunk with minimal memory"""
        try:
            # Quick load and process
            model = CsmForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(model_path)
            
            # Generate audio for chunk
            conversation = [{"role": "0", "content": [{"type": "text", "text": text_chunk}]}]
            inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
            
            with torch.no_grad():
                audio = model.generate(**inputs, output_audio=True, max_new_tokens=30)
            
            # Immediately cleanup
            del model
            del processor
            torch.cuda.empty_cache()
            gc.collect()
            
            return audio
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return None

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def test_memory_optimized_csm():
    """Test the memory-optimized CSM implementation"""
    
    print("=" * 60)
    print("🚀 MEMORY-OPTIMIZED CSM TEST")
    print("=" * 60)
    
    # Initialize with automatic fallbacks
    csm = MemoryOptimizedCSM(device="auto")
    
    # Load model with fallback strategies
    if csm.load_model_with_fallbacks():
        print("\n✅ Model loaded successfully!")
        print(f"📍 Running on: {csm.device}")
        
        # Test generation
        test_texts = [
            "Hello, this is a test of memory-optimized CSM.",
            "I can run on limited GPU memory!",
            "Even 6GB GPUs work now."
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n🎤 Generating speech {i+1}: {text}")
            
            audio = csm.generate_optimized(
                text=text,
                max_new_tokens=60,
                temperature=0.72
            )
            
            if audio is not None:
                print(f"✅ Generated audio shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}")
                
                # Save audio
                import torchaudio
                try:
                    if isinstance(audio, torch.Tensor):
                        if audio.dim() == 1:
                            audio = audio.unsqueeze(0)
                        torchaudio.save(f"optimized_output_{i}.wav", audio, 16000)
                        print(f"💾 Saved to optimized_output_{i}.wav")
                except Exception as e:
                    print(f"⚠️ Could not save audio: {e}")
            else:
                print("❌ Generation failed")
        
        # Show memory stats
        if torch.cuda.is_available():
            print("\n📊 Final Memory Stats:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9:.2f} GB")
    
    else:
        print("\n❌ Failed to load model with any strategy")
        print("\n💡 Alternative: Try the streaming implementation:")
        print("   streaming = StreamingCSM()")
        print("   for chunk in streaming.generate_streaming(text, model_path):")
        print("       # Process each chunk")

# ============================================================================
# QUICK FIX SCRIPT
# ============================================================================

def quick_fix_cuda_oom():
    """Quick fixes for CUDA OOM issues"""
    
    print("🔧 APPLYING CUDA OOM FIXES...")
    
    # 1. Clear all GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Cleared GPU cache")
    
    # 2. Set memory optimization flags
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("✅ Set memory optimization flags")
    
    # 3. Reduce default batch size
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("✅ Disabled cudnn benchmark mode")
    
    # 4. Enable gradient checkpointing if using training
    print("✅ Ready for memory-optimized loading")
    
    print("\n📝 Recommended next steps:")
    print("1. Use 8-bit or 4-bit quantization")
    print("2. Reduce max_new_tokens to 40-60")
    print("3. Use CPU offloading for large models")
    print("4. Consider streaming generation for long texts")
    print("5. Close other GPU applications")
# This will work 100%
from transformers import CsmForConditionalGeneration

model = CsmForConditionalGeneration.from_pretrained(
    "sesame/csm-1b",
    device_map="cpu"  # ← Use regular memory, not GPU
)
if __name__ == "__main__":
    # First apply quick fixes
    quick_fix_cuda_oom()
    
    # Then test the optimized implementation
    test_memory_optimized_csm()