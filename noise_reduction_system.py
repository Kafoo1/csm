import torch
import torchaudio
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import librosa

class AdvancedNoiseReducer:
    """Advanced noise reduction for CSM-generated audio"""
    
    def __init__(self):
        self.sample_rate = 24000  # CSM default
        
    def reduce_background_noise(self, audio_file, output_file):
        """Main noise reduction function"""
        
        print(f'🔧 Reducing noise in {audio_file}...')
        
        # Load audio
        waveform, sr = torchaudio.load(audio_file)
        
        # Convert to numpy for processing
        audio_np = waveform.numpy().squeeze()
        
        # Apply multiple noise reduction techniques
        cleaned_audio = self._multi_stage_noise_reduction(audio_np, sr)
        
        # Convert back to tensor
        cleaned_tensor = torch.tensor(cleaned_audio).unsqueeze(0)
        
        # Save cleaned audio
        torchaudio.save(output_file, cleaned_tensor, sr)
        print(f'✅ Clean audio saved: {output_file}')
        
        return output_file
    
    def _multi_stage_noise_reduction(self, audio, sr):
        """Multi-stage noise reduction pipeline"""
        
        # Stage 1: Spectral gating (remove constant background noise)
        audio = self._spectral_gating(audio, sr)
        
        # Stage 2: High-pass filter (remove low-frequency rumble)
        audio = self._high_pass_filter(audio, sr, cutoff=80)
        
        # Stage 3: Noise gate (remove quiet artifacts)
        audio = self._noise_gate(audio, threshold=0.02)
        
        # Stage 4: Spectral subtraction
        audio = self._spectral_subtraction(audio, sr)
        
        # Stage 5: Gentle normalization
        audio = self._normalize_audio(audio)
        
        return audio
    
    def _spectral_gating(self, audio, sr):
        """Remove constant background noise using spectral gating"""
        
        # Use librosa for spectral processing
        try:
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Compute spectrogram
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise profile from quiet sections
            noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
            
            # Create noise gate
            noise_reduction_factor = 0.1  # Reduce noise by 90%
            gate = np.maximum(magnitude - noise_profile * 2, 
                            magnitude * noise_reduction_factor) / magnitude
            
            # Apply gate
            cleaned_magnitude = magnitude * gate
            
            # Reconstruct audio
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft, hop_length=512)
            
            return cleaned_audio
            
        except Exception as e:
            print(f'⚠️ Spectral gating failed: {e}, using original audio')
            return audio
    
    def _high_pass_filter(self, audio, sr, cutoff=80):
        """Remove low-frequency noise with high-pass filter"""
        
        try:
            nyquist = sr / 2
            normalized_cutoff = cutoff / nyquist
            
            b, a = butter(4, normalized_cutoff, btype='high')
            filtered_audio = filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            print(f'⚠️ High-pass filter failed: {e}')
            return audio
    
    def _noise_gate(self, audio, threshold=0.02):
        """Simple noise gate to remove quiet artifacts"""
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        smoothed_envelope = gaussian_filter1d(envelope, sigma=10)
        
        # Create gate
        gate = (smoothed_envelope > threshold).astype(float)
        
        # Apply soft gating to avoid clicks
        gate = gaussian_filter1d(gate, sigma=5)
        
        return audio * gate
    
    def _spectral_subtraction(self, audio, sr):
        """Advanced spectral subtraction for noise reduction"""
        
        try:
            # Estimate noise from first 0.5 seconds (usually silence/noise)
            noise_duration = min(int(0.5 * sr), len(audio) // 4)
            noise_sample = audio[:noise_duration]
            
            # Compute spectrograms
            stft_audio = librosa.stft(audio, n_fft=1024, hop_length=256)
            stft_noise = librosa.stft(noise_sample, n_fft=1024, hop_length=256)
            
            # Get magnitude and phase
            mag_audio = np.abs(stft_audio)
            phase_audio = np.angle(stft_audio)
            mag_noise = np.mean(np.abs(stft_noise), axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.1   # Spectral floor
            
            mag_clean = mag_audio - alpha * mag_noise
            mag_clean = np.maximum(mag_clean, beta * mag_audio)
            
            # Reconstruct
            stft_clean = mag_clean * np.exp(1j * phase_audio)
            audio_clean = librosa.istft(stft_clean, hop_length=256)
            
            return audio_clean
            
        except Exception as e:
            print(f'⚠️ Spectral subtraction failed: {e}')
            return audio
    
    def _normalize_audio(self, audio, target_level=0.9):
        """Gentle normalization to prevent clipping"""
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * target_level
        
        return audio

# Integration with CSM system
def process_csm_audio_with_noise_reduction(csm_audio, filename_base):
    """Process CSM audio with noise reduction"""
    
    # Save original CSM audio
    original_file = f"{filename_base}_original.wav"
    processor.save_audio(csm_audio, original_file)
    
    # Apply noise reduction
    noise_reducer = AdvancedNoiseReducer()
    clean_file = f"{filename_base}_clean.wav"
    noise_reducer.reduce_background_noise(original_file, clean_file)
    
    return clean_file

# Updated voice generation with noise reduction
def generate_clean_voice(universal_agent, phone_number, response_text):
    """Generate voice with automatic noise reduction"""
    
    # Generate audio using universal agent
    audio, text = universal_agent.generate_voice_response(phone_number, response_text)
    
    if audio:
        # Apply noise reduction
        clean_file = process_csm_audio_with_noise_reduction(audio, "clean_response")
        print(f'🎧 Clean audio ready: {clean_file}')
        return clean_file, text
    
    return None, text

# Test noise reduction
def test_noise_reduction():
    """Test noise reduction on existing files"""
    
    print('🧪 Testing noise reduction...')
    
    noise_reducer = AdvancedNoiseReducer()
    
    # Test files (replace with your actual files)
    test_files = [
        'maya_safe_greeting.wav',
        'maya_safe_menu.wav', 
        'maya_safe_order.wav'
    ]
    
    for file in test_files:
        try:
            clean_file = file.replace('.wav', '_CLEAN.wav')
            noise_reducer.reduce_background_noise(file, clean_file)
        except Exception as e:
            print(f'❌ Failed to clean {file}: {e}')
    
    print('✅ Noise reduction testing complete!')

if __name__ == "__main__":
    test_noise_reduction()