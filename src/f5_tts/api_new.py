import random
import sys
import os
import tempfile
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import re
from collections import OrderedDict
import json
import argparse
from datetime import datetime

import soundfile as sf
import torchaudio
from cached_path import cached_path

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat
from f5_tts.utils.whisper_api import translate_inference, transribe_inference
import torch


class F5TTSThaiAPI:
    def __init__(
        self,
        model_type="Default",
        custom_model_path=None,
        vocab_path="./vocab/vocab.txt",
        device=None,
        hf_cache_dir=None,
        profiles_dir="./profiles"
    ):
        """
        Initialize F5-TTS Thai API
        
        Args:
            model_type: "Default", "FP16", or "Custom"
            custom_model_path: Path to custom model if model_type is "Custom"
            vocab_path: Path to vocabulary file
            device: Device to run on (auto-detect if None)
            hf_cache_dir: Cache directory for hugging face models
            profiles_dir: Directory to store audio profiles
        """
        # Model paths
        self.default_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt"
        self.fp16_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_650000_FP16.pt"
        self.vocab_base = vocab_path
        
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.hf_cache_dir = hf_cache_dir
        self.profiles_dir = profiles_dir
        
        # Create profiles directory
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Initialize profiles storage
        self.profiles = self._load_profiles()
        
        # Set device
        if device is not None:
            self.device = device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        
        # Load vocoder
        self.vocoder = load_vocoder()
        
        # Load model
        self.f5tts_model = None
        self.load_model(model_type, custom_model_path)
    
    def load_model(self, model_type="Default", custom_model_path=None):
        """Load F5-TTS model"""
        torch.cuda.empty_cache()
        
        if model_type == "Custom" and custom_model_path:
            model_path = custom_model_path
        elif model_type == "FP16":
            model_path = self.fp16_model_base
        else:
            model_path = self.default_model_base
        
        # Load model configuration
        # Load model configuration
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)
        
        # Determine vocab file path
        vocab_file = self.vocab_base if os.path.exists(self.vocab_base) else str(
            cached_path("hf://VIZINTZOR/F5-TTS-THAI/vocab.txt", cache_dir=self.hf_cache_dir)
        )
        
        # Load the model
        self.f5tts_model = load_model(
            DiT, 
            F5TTS_model_cfg, 
            str(cached_path(model_path, cache_dir=self.hf_cache_dir)), 
            vocab_file=vocab_file, 
            use_ema=True
        )
        
        print(f"Loaded model: {model_type} from {model_path}")
        return f"Loaded Model {model_type}"
    
    def infer_tts(
        self,
        ref_audio_path: Optional[str],
        ref_text: str,
        gen_text: str,
        remove_silence: bool = True,
        cross_fade_duration: float = 0.15,
        nfe_step: int = 32,
        speed: float = 1.0,
        cfg_strength: float = 2.0,
        max_chars: int = 250,
        seed: int = -1,
        no_ref_audio: bool = False,
        output_path: Optional[str] = None,
        return_spectrogram: bool = False,
        use_ipa: bool = False
    ) -> Dict[str, Any]:
        """
        Generate speech using F5-TTS
        
        Args:
            ref_audio_path: Path to reference audio file (can be None for no_ref_audio)
            ref_text: Reference text (transcription of reference audio)
            gen_text: Text to generate speech for
            remove_silence: Whether to remove silence from generated audio
            cross_fade_duration: Duration for cross-fading between segments
            nfe_step: Number of NFE steps (higher = better quality, slower)
            speed: Speech speed multiplier
            cfg_strength: CFG strength
            max_chars: Maximum characters per segment for long text
            seed: Random seed (-1 for random)
            output_path: Path to save output audio file
            return_spectrogram: Whether to return spectrogram data
            
        Returns:
            Dictionary containing:
            - audio_data: numpy array of audio data
            - sample_rate: sample rate
            - seed: used seed
            - ref_text: processed reference text
            - spectrogram_path: path to spectrogram image (if return_spectrogram=True)
        """
        if self.f5tts_model is None:
            raise RuntimeError("Model not loaded. Please load a model first.")
        
        if not ref_audio_path and not ref_text:
            raise ValueError("Either reference audio path or reference text is required")
        
        if not gen_text.strip():
            raise ValueError("Generation text cannot be empty")
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        
        # Preprocess reference audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        
        # Clean generation text
        gen_text_cleaned = process_thai_repeat(replace_numbers_with_thai(gen_text))
        
        # Generate speech
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            gen_text_cleaned,
            self.f5tts_model,
            self.vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            cfg_strength=cfg_strength,
            target_rms=0.1,
            sway_sampling_coef=-1,
            set_max_chars=max_chars,
            device=self.device,
            use_ipa=use_ipa
        )
        
        # Remove silence if requested
        if remove_silence:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()
        
        # Save output audio if path provided
        if output_path:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            sf.write(output_path, final_wave, final_sample_rate)
        
        # Prepare result
        result = {
            "audio_data": final_wave,
            "sample_rate": final_sample_rate,
            "seed": seed,
            "ref_text": ref_text
        }
        
        # Save spectrogram if requested
        if return_spectrogram:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
                spectrogram_path = tmp_spectrogram.name
                save_spectrogram(combined_spectrogram, spectrogram_path)
                result["spectrogram_path"] = spectrogram_path
        
        return result
    
    def parse_speechtypes_text(self, gen_text: str) -> List[Dict[str, str]]:
        """
        Parse text with speech types marked as {style}
        
        Args:
            gen_text: Text with speech types like "{normal} Hello {sad} I'm sad"
            
        Returns:
            List of segments with style and text
        """
        pattern = r"\{(.*?)\}"
        tokens = re.split(pattern, gen_text)
        
        segments = []
        current_style = "Regular"
        
        for i in range(len(tokens)):
            if i % 2 == 0:
                # This is text
                text = tokens[i].strip()
                if text:
                    segments.append({"style": current_style, "text": text})
            else:
                # This is style
                style = tokens[i].strip()
                current_style = style
        
        return segments
    
    def infer_multistyle(
        self,
        gen_text: str,
        speech_types: Dict[str, Dict[str, str]],
        remove_silence: bool = True,
        cross_fade_duration: float = 0.15,
        nfe_step: int = 32,
        speed: float = 1.0,
        cfg_strength: float = 2.0,
        max_chars: int = 250,
        seed: int = -1,
        output_path: Optional[str] = None,
        return_spectrogram: bool = False
    ) -> Dict[str, Any]:
        """
        Generate multistyle speech
        
        Args:
            gen_text: Text with style markers like "{normal} Hello {sad} I'm sad"
            speech_types: Dict with style names as keys and dict containing 'audio' and 'ref_text' as values
            remove_silence: Whether to remove silence
            cross_fade_duration: Cross fade duration
            nfe_step: NFE steps
            speed: Speech speed
            cfg_strength: CFG strength
            max_chars: Max characters per segment
            seed: Random seed
            output_path: Output file path
            return_spectrogram: Whether to return spectrogram data
            
        Returns:
            Dictionary with generated audio data and metadata
        """
        if self.f5tts_model is None:
            raise RuntimeError("Model not loaded. Please load a model first.")
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        
        # Parse text segments
        segments = self.parse_speechtypes_text(gen_text)
        
        # Generate audio for each segment
        generated_audio_segments = []
        combined_spectrograms = []
        current_style = "Regular"
        
        for segment in segments:
            style = segment["style"]
            text = segment["text"]
            
            if style in speech_types:
                current_style = style
            else:
                print(f"Warning: Style {style} not found, using Regular as default")
                current_style = "Regular"
            
            if current_style not in speech_types:
                raise ValueError(f"Reference audio for style {current_style} not provided")
            
            ref_audio_path = speech_types[current_style]["audio"]
            ref_text = speech_types[current_style].get("ref_text", "")
            
            # Generate speech for this segment
            result = self.infer_tts(
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                gen_text=text,
                remove_silence=remove_silence,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                speed=speed,
                cfg_strength=cfg_strength,
                max_chars=max_chars,
                seed=seed,
                no_ref_audio=False,
                return_spectrogram=return_spectrogram
            )
            
            generated_audio_segments.append(result["audio_data"])
            
            # Collect spectrograms if requested
            if return_spectrogram and 'spectrogram_path' in result:
                combined_spectrograms.append(result['spectrogram_path'])
                
            # Update ref_text for this style
            speech_types[current_style]["ref_text"] = result["ref_text"]
        
        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            
            # Save output if path provided
            if output_path:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                sf.write(output_path, final_audio_data, self.target_sample_rate)
            
            return {
                "audio_data": final_audio_data,
                "sample_rate": self.target_sample_rate,
                "seed": seed,
                "speech_types": speech_types,
                "spectrogram_paths": combined_spectrograms if return_spectrogram else None
            }
        else:
            raise RuntimeError("No audio generated")
    
    def transcribe_audio(
        self,
        audio_path: str,
        translate: bool = False,
        model: str = "large-v3-turbo",
        compute_type: str = "auto",
        target_language: str = "th",
        source_language: str = "th"
    ) -> str:
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_path: Path to audio file
            translate: Whether to translate the text
            model: Whisper model to use
            compute_type: Compute type for inference ("auto", "float16", "float32", "int8")
            target_language: Target language for translation
            source_language: Source language
            
        Returns:
            Transcribed/translated text
        """
        # Auto-detect compute type based on device and platform
        if compute_type == "auto":
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                compute_type = "float32"  # Mac works best with float32
            elif self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "float32"  # CPU fallback
        
        # Try different compute types in order of preference for Mac
        compute_types_to_try = []
        if compute_type == "float32":
            compute_types_to_try = ["float32", "int8"]
        elif compute_type == "float16":
            compute_types_to_try = ["float16", "float32", "int8"]
        else:
            compute_types_to_try = [compute_type, "float32", "int8"]
        
        last_error = None
        for ct in compute_types_to_try:
            try:
                print(f"Trying transcription with compute_type: {ct}")
                if translate:
                    output_text = translate_inference(
                        text=transribe_inference(
                            input_audio=audio_path,
                            model=model,
                            compute_type=ct,
                            language=source_language
                        ),
                        target=target_language
                    )
                else:
                    output_text = transribe_inference(
                        input_audio=audio_path,
                        model=model,
                        compute_type=ct,
                        language=source_language
                    )
                print(f"Transcription successful with compute_type: {ct}")
                return output_text
            except Exception as e:
                last_error = e
                print(f"Transcription failed with compute_type {ct}: {e}")
                continue
        
        # If all attempts failed, try using the original transcribe function from utils_infer
        try:
            print("Trying fallback transcription method...")
            output_text = transcribe(audio_path, language=source_language)
            print("Fallback transcription successful")
            return output_text
        except Exception as e:
            print(f"Fallback transcription failed: {e}")
        
        # If everything failed, raise the last error
        raise RuntimeError(f"All transcription methods failed. Last error: {last_error}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            "device": self.device,
            "target_sample_rate": self.target_sample_rate,
            "hop_length": self.hop_length,
            "model_loaded": self.f5tts_model is not None
        }
    
    def _ensure_output_dir(self, file_path: str) -> str:
        """Ensure output directory exists and return the path"""
        if file_path:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        return file_path
    
    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load profiles from disk"""
        profiles_file = os.path.join(self.profiles_dir, "profiles.json")
        if os.path.exists(profiles_file):
            try:
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading profiles: {e}")
                return {}
        return {}
    
    def _save_profiles(self):
        """Save profiles to disk"""
        profiles_file = os.path.join(self.profiles_dir, "profiles.json")
        try:
            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving profiles: {e}")
    
    def create_profile(
        self,
        profile_name: str,
        ref_audio_path: str,
        ref_text: str,
        description: str = "",
        emotion: str = "normal",
        overwrite: bool = False
    ) -> bool:
        """
        Create a new audio profile
        
        Args:
            profile_name: Name for the profile
            ref_audio_path: Path to reference audio file
            ref_text: Reference text
            description: Profile description
            emotion: Emotion/mood of the audio (e.g., "normal", "happy", "sad", "angry", "calm")
            overwrite: Whether to overwrite existing profile
            
        Returns:
            Success status
        """
        if profile_name in self.profiles and not overwrite:
            raise ValueError(f"Profile '{profile_name}' already exists. Use overwrite=True to replace it.")
        
        # Copy audio file to profiles directory
        audio_filename = f"{profile_name}.wav"
        profile_audio_path = os.path.join(self.profiles_dir, audio_filename)
        
        try:
            # Load and save audio to ensure consistent format
            import shutil
            shutil.copy2(ref_audio_path, profile_audio_path)
            
            # Create profile entry
            self.profiles[profile_name] = {
                "audio_path": profile_audio_path,
                "ref_text": ref_text,
                "description": description,
                "emotion": emotion,
                "created_at": datetime.now().isoformat()
            }
            
            self._save_profiles()
            print(f"Profile '{profile_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating profile: {e}")
            return False
    
    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a profile
        
        Args:
            profile_name: Name of profile to delete
            
        Returns:
            Success status
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        try:
            # Remove audio file
            audio_path = self.profiles[profile_name]["audio_path"]
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Remove from profiles
            del self.profiles[profile_name]
            self._save_profiles()
            print(f"Profile '{profile_name}' deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting profile: {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """List available profiles"""
        return list(self.profiles.keys())
    
    def get_profile_info(self, profile_name: str) -> Dict[str, Any]:
        """
        Get profile information
        
        Args:
            profile_name: Name of profile
            
        Returns:
            Profile information
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        profile = self.profiles[profile_name].copy()
        # Add existence check for audio file
        profile["audio_exists"] = os.path.exists(profile["audio_path"])
        return profile
    
    def get_profiles_by_emotion(self, emotion: str) -> List[str]:
        """
        Get profiles filtered by emotion
        
        Args:
            emotion: Emotion to filter by
            
        Returns:
            List of profile names with matching emotion
        """
        return [name for name, profile in self.profiles.items() 
                if profile.get("emotion", "normal") == emotion]
    
    def get_profile_emotions(self) -> Dict[str, List[str]]:
        """
        Get all profiles grouped by emotion
        
        Returns:
            Dictionary with emotions as keys and list of profile names as values
        """
        emotions = {}
        for name, profile in self.profiles.items():
            emotion = profile.get("emotion", "normal")
            if emotion not in emotions:
                emotions[emotion] = []
            emotions[emotion].append(name)
        return emotions
    
    def infer_tts_with_profile(
        self,
        profile_name: str,
        gen_text: str,
        remove_silence: bool = True,
        cross_fade_duration: float = 0.15,
        nfe_step: int = 32,
        speed: float = 1.0,
        cfg_strength: float = 2.0,
        max_chars: int = 250,
        seed: int = -1,
        no_ref_audio: bool = False,
        output_path: Optional[str] = None,
        return_spectrogram: bool = False,
        use_ipa: bool = False
    ) -> Dict[str, Any]:
        """
        Generate speech using a stored profile
        
        Args:
            profile_name: Name of profile to use
            gen_text: Text to generate speech for
            ... (other parameters same as infer_tts)
            
        Returns:
            Dictionary with generated audio data and metadata
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        profile = self.profiles[profile_name]
        
        return self.infer_tts(
            ref_audio_path=profile["audio_path"],
            ref_text=profile["ref_text"],
            gen_text=gen_text,
            remove_silence=remove_silence,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            cfg_strength=cfg_strength,
            max_chars=max_chars,
            seed=seed,
            no_ref_audio=no_ref_audio,
            output_path=output_path,
            return_spectrogram=return_spectrogram,
            use_ipa=use_ipa
        )
    
    def infer_multistyle_with_profiles(
        self,
        gen_text: str,
        profile_emotions: Dict[str, str],
        remove_silence: bool = True,
        cross_fade_duration: float = 0.15,
        nfe_step: int = 32,
        speed: float = 1.0,
        cfg_strength: float = 2.0,
        max_chars: int = 250,
        seed: int = -1,
        output_path: Optional[str] = None,
        return_spectrogram: bool = False
    ) -> Dict[str, Any]:
        """
        Generate multistyle speech using stored profiles
        
        Args:
            gen_text: Text with style markers like "{normal} Hello {sad} I'm sad"
            profile_emotions: Dict mapping style names to profile names or emotions
            ... (other parameters same as infer_multistyle)
            
        Returns:
            Dictionary with generated audio data and metadata
        """
        # Convert profile_emotions to speech_types format
        speech_types = {}
        
        for style_name, profile_identifier in profile_emotions.items():
            # Check if it's a direct profile name
            if profile_identifier in self.profiles:
                profile = self.profiles[profile_identifier]
                speech_types[style_name] = {
                    "audio": profile["audio_path"],
                    "ref_text": profile["ref_text"]
                }
            else:
                # Try to find profile by emotion
                matching_profiles = self.get_profiles_by_emotion(profile_identifier)
                if matching_profiles:
                    # Use the first matching profile
                    profile = self.profiles[matching_profiles[0]]
                    speech_types[style_name] = {
                        "audio": profile["audio_path"],
                        "ref_text": profile["ref_text"]
                    }
                else:
                    raise ValueError(f"No profile found for '{profile_identifier}' (style: {style_name})")
        
        # Use the regular multistyle inference
        return self.infer_multistyle(
            gen_text=gen_text,
            speech_types=speech_types,
            remove_silence=remove_silence,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            cfg_strength=cfg_strength,
            max_chars=max_chars,
            seed=seed,
            output_path=output_path,
            return_spectrogram=return_spectrogram
        )


# REST API Server
try:
    from flask import Flask, request, jsonify, send_file
    from werkzeug.utils import secure_filename
    import base64
    import io
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. REST API server will be disabled.")

class F5TTSAPIServer:
    def __init__(self, api_instance: F5TTSThaiAPI, host='0.0.0.0', port=4000):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for REST API server")
        
        self.api = api_instance
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        self._setup_routes()
    
    def _parse_bool(self, value, default=False):
        """Helper function to handle boolean values from JSON or form data"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return default
    
    def _handle_audio_file(self, audio_data, filename_prefix="uploaded_audio"):
        """Handle uploaded audio file and return temporary path"""
        if audio_data is None:
            return None
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save uploaded audio
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        
        return temp_path
    
    def _cleanup_temp_file(self, temp_path):
        """Clean up temporary file and directory"""
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                # Remove temporary directory if empty
                temp_dir = os.path.dirname(temp_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.api.f5tts_model is not None,
                'device': self.api.device
            })
        
        @self.app.route('/tts', methods=['POST'])
        def text_to_speech():
            """Text to speech endpoint"""
            try:
                # Handle both JSON and form-data
                if request.is_json:
                    data = request.get_json()
                    ref_audio_data = None
                else:
                    # Handle form-data with file uploads
                    data = request.form.to_dict()
                    ref_audio_data = request.files.get('ref_audio')
                
                # Check return format preference
                return_format = data.get('return_format', 'audio_base64')  # Default to base64
                
                # Handle audio input - either file upload or path
                ref_audio_path = None
                temp_audio_path = None
                
                if ref_audio_data:
                    # Handle uploaded file
                    temp_audio_path = self._handle_audio_file(ref_audio_data.read(), "ref_audio")
                    ref_audio_path = temp_audio_path
                elif 'ref_audio_path' in data:
                    # Handle file path
                    ref_audio_path = data['ref_audio_path']
                elif 'ref_audio' in data:
                    # Handle base64 encoded audio
                    try:
                        audio_bytes = base64.b64decode(data['ref_audio'])
                        temp_audio_path = self._handle_audio_file(audio_bytes, "ref_audio")
                        ref_audio_path = temp_audio_path
                    except Exception as e:
                        return jsonify({'success': False, 'error': f'Invalid base64 audio data: {str(e)}'}), 400
                
                # Check if using profile
                if 'profile_name' in data:
                    result = self.api.infer_tts_with_profile(
                        profile_name=data['profile_name'],
                        gen_text=data['gen_text'],
                        remove_silence=self._parse_bool(data.get('remove_silence', True)),
                        cross_fade_duration=float(data.get('cross_fade_duration', 0.15)),
                        nfe_step=int(data.get('nfe_step', 32)),
                        speed=float(data.get('speed', 1.0)),
                        cfg_strength=float(data.get('cfg_strength', 2.0)),
                        max_chars=int(data.get('max_chars', 250)),
                        seed=int(data.get('seed', -1)),
                        no_ref_audio=self._parse_bool(data.get('no_ref_audio', False)),
                        return_spectrogram=self._parse_bool(data.get('return_spectrogram', False))
                    )
                else:
                    if not ref_audio_path:
                        return jsonify({'success': False, 'error': 'ref_audio_path, ref_audio file, or profile_name is required'}), 400
                    
                    result = self.api.infer_tts(
                        ref_audio_path=ref_audio_path,
                        ref_text=data.get('ref_text', ''),
                        gen_text=data['gen_text'],
                        remove_silence=self._parse_bool(data.get('remove_silence', True)),
                        cross_fade_duration=float(data.get('cross_fade_duration', 0.15)),
                        nfe_step=int(data.get('nfe_step', 32)),
                        speed=float(data.get('speed', 1.0)),
                        cfg_strength=float(data.get('cfg_strength', 2.0)),
                        max_chars=int(data.get('max_chars', 250)),
                        seed=int(data.get('seed', -1)),
                        no_ref_audio=self._parse_bool(data.get('no_ref_audio', False)),
                        return_spectrogram=self._parse_bool(data.get('return_spectrogram', False))
                    )
                
                # Prepare response based on return format
                response = {
                    'success': True,
                    'sample_rate': result['sample_rate'],
                    'seed': result['seed'],
                    'ref_text': result['ref_text']
                }
                
                # Handle audio output based on return_format
                if return_format == 'audio_file':
                    # Create temporary file for audio
                    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_audio_path = temp_audio_file.name
                    temp_audio_file.close()
                    
                    # Save audio to temporary file
                    sf.write(temp_audio_path, result['audio_data'], result['sample_rate'])
                    
                    # Return file response
                    response['audio_file'] = temp_audio_path
                    response['message'] = 'Audio file generated successfully'
                    
                    # Clean up temp file after a delay (or let client handle cleanup)
                    return send_file(
                        temp_audio_path, 
                        as_attachment=True, 
                        download_name=f"tts_output_{result['seed']}.wav",
                        mimetype='audio/wav'
                    )
                else:
                    # Return base64 encoded audio (default)
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, result['audio_data'], result['sample_rate'], format='WAV')
                    audio_buffer.seek(0)
                    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                    
                    response['audio_base64'] = audio_base64
                
                # Handle spectrogram if requested
                if 'spectrogram_path' in result:
                    with open(result['spectrogram_path'], 'rb') as f:
                        spec_base64 = base64.b64encode(f.read()).decode('utf-8')
                        response['spectrogram_base64'] = spec_base64
                
                # Cleanup temporary file
                self._cleanup_temp_file(temp_audio_path)
                
                # Return JSON response only for base64 format
                if return_format != 'audio_file':
                    return jsonify(response)
                
            except Exception as e:
                # Cleanup temporary file in case of error
                if 'temp_audio_path' in locals():
                    self._cleanup_temp_file(temp_audio_path)
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/multistyle', methods=['POST'])
        def multistyle_tts():
            """Multistyle TTS endpoint with support for Direct, File Upload, and Profile modes"""
            try:
                # Handle both JSON and form-data
                if request.is_json:
                    data = request.get_json()
                    mode = data.get('mode', 'direct')  # 'direct', 'file_upload', 'profile'
                else:
                    # Handle form-data with file uploads
                    data = request.form.to_dict()
                    mode = data.get('mode', 'file_upload')
                
                # Check return format preference
                return_format = data.get('return_format', 'audio_base64')  # Default to base64
                
                # Handle different modes
                if mode == 'profile':
                    # Profile mode - use stored profiles
                    profile_emotions = json.loads(data.get('profile_emotions', '{}'))
                    
                    result = self.api.infer_multistyle_with_profiles(
                        gen_text=data['gen_text'],
                        profile_emotions=profile_emotions,
                        remove_silence=self._parse_bool(data.get('remove_silence', True)),
                        cross_fade_duration=float(data.get('cross_fade_duration', 0.15)),
                        nfe_step=int(data.get('nfe_step', 32)),
                        speed=float(data.get('speed', 1.0)),
                        cfg_strength=float(data.get('cfg_strength', 2.0)),
                        max_chars=int(data.get('max_chars', 250)),
                        seed=int(data.get('seed', -1)),
                        return_spectrogram=self._parse_bool(data.get('return_spectrogram', False))
                    )
                    
                elif mode == 'file_upload':
                    # File upload mode - handle uploaded files
                    speech_types = json.loads(data.get('speech_types', '{}'))
                    
                    # Handle uploaded audio files for speech types
                    for style_name in speech_types:
                        audio_file = request.files.get(f'audio_{style_name}')
                        if audio_file:
                            temp_path = self._handle_audio_file(audio_file.read(), f"style_{style_name}")
                            speech_types[style_name]['audio'] = temp_path
                    
                    result = self.api.infer_multistyle(
                        gen_text=data['gen_text'],
                        speech_types=speech_types,
                        remove_silence=self._parse_bool(data.get('remove_silence', True)),
                        cross_fade_duration=float(data.get('cross_fade_duration', 0.15)),
                        nfe_step=int(data.get('nfe_step', 32)),
                        speed=float(data.get('speed', 1.0)),
                        cfg_strength=float(data.get('cfg_strength', 2.0)),
                        max_chars=int(data.get('max_chars', 250)),
                        seed=int(data.get('seed', -1)),
                        return_spectrogram=self._parse_bool(data.get('return_spectrogram', False))
                    )
                    
                else:
                    # Direct mode - use file paths or base64
                    if request.is_json:
                        speech_types = data['speech_types']
                    else:
                        speech_types = json.loads(data.get('speech_types', '{}'))
                    
                    # Handle base64 encoded audio if present
                    for style_name in speech_types:
                        if f'audio_base64_{style_name}' in data:
                            try:
                                audio_bytes = base64.b64decode(data[f'audio_base64_{style_name}'])
                                temp_path = self._handle_audio_file(audio_bytes, f"style_{style_name}")
                                speech_types[style_name]['audio'] = temp_path
                            except Exception as e:
                                return jsonify({'success': False, 'error': f'Invalid base64 audio data for {style_name}: {str(e)}'}), 400
                    
                    result = self.api.infer_multistyle(
                        gen_text=data['gen_text'],
                        speech_types=speech_types,
                        remove_silence=self._parse_bool(data.get('remove_silence', True)),
                        cross_fade_duration=float(data.get('cross_fade_duration', 0.15)),
                        nfe_step=int(data.get('nfe_step', 32)),
                        speed=float(data.get('speed', 1.0)),
                        cfg_strength=float(data.get('cfg_strength', 2.0)),
                        max_chars=int(data.get('max_chars', 250)),
                        seed=int(data.get('seed', -1)),
                        return_spectrogram=self._parse_bool(data.get('return_spectrogram', False))
                    )
                
                # Prepare response based on return format
                response = {
                    'success': True,
                    'sample_rate': result['sample_rate'],
                    'seed': result['seed'],
                    'speech_types': result['speech_types']
                }
                
                # Handle audio output based on return_format
                if return_format == 'audio_file':
                    # Create temporary file for audio
                    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_audio_path = temp_audio_file.name
                    temp_audio_file.close()
                    
                    # Save audio to temporary file
                    sf.write(temp_audio_path, result['audio_data'], result['sample_rate'])
                    
                    # Cleanup temporary files
                    if not request.is_json:
                        for style_name in speech_types:
                            temp_path = speech_types[style_name].get('audio', '')
                            if temp_path and temp_path.startswith(tempfile.gettempdir()):
                                self._cleanup_temp_file(temp_path)
                    
                    # Return file response
                    return send_file(
                        temp_audio_path, 
                        as_attachment=True, 
                        download_name=f"multistyle_output_{result['seed']}.wav",
                        mimetype='audio/wav'
                    )
                else:
                    # Return base64 encoded audio (default)
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, result['audio_data'], result['sample_rate'], format='WAV')
                    audio_buffer.seek(0)
                    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                    
                    response['audio_base64'] = audio_base64
                
                # Handle spectrograms if requested
                if 'spectrogram_paths' in result and result['spectrogram_paths']:
                    spectrograms_base64 = []
                    for spec_path in result['spectrogram_paths']:
                        if spec_path and os.path.exists(spec_path):
                            with open(spec_path, 'rb') as f:
                                spec_base64 = base64.b64encode(f.read()).decode('utf-8')
                                spectrograms_base64.append(spec_base64)
                    response['spectrograms_base64'] = spectrograms_base64
                
                # Cleanup temporary files
                if mode == 'file_upload' and not request.is_json:
                    for style_name in speech_types:
                        temp_path = speech_types[style_name].get('audio', '')
                        if temp_path and temp_path.startswith(tempfile.gettempdir()):
                            self._cleanup_temp_file(temp_path)
                elif mode == 'direct':
                    # Cleanup base64 temporary files
                    for style_name in speech_types:
                        temp_path = speech_types[style_name].get('audio', '')
                        if temp_path and temp_path.startswith(tempfile.gettempdir()):
                            self._cleanup_temp_file(temp_path)
                
                # Return JSON response only for base64 format
                if return_format != 'audio_file':
                    return jsonify(response)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/transcribe', methods=['POST'])
        def transcribe():
            """Audio transcription endpoint"""
            try:
                # Handle both JSON and form-data
                if request.is_json:
                    data = request.get_json()
                    audio_path = data['audio_path']
                    temp_audio_path = None
                else:
                    # Handle form-data with file upload
                    data = request.form.to_dict()
                    audio_file = request.files.get('audio_file')
                    
                    if audio_file:
                        temp_audio_path = self._handle_audio_file(audio_file.read(), "transcribe_audio")
                        audio_path = temp_audio_path
                    elif 'audio_path' in data:
                        audio_path = data['audio_path']
                        temp_audio_path = None
                    else:
                        return jsonify({'success': False, 'error': 'audio_file or audio_path is required'}), 400
                
                result = self.api.transcribe_audio(
                    audio_path=audio_path,
                    translate=self._parse_bool(data.get('translate', False)),
                    model=data.get('model', 'large-v3-turbo'),
                    compute_type=data.get('compute_type', 'auto'),
                    target_language=data.get('target_language', 'th'),
                    source_language=data.get('source_language', 'th')
                )
                
                # Cleanup temporary file
                self._cleanup_temp_file(temp_audio_path)
                
                return jsonify({
                    'success': True,
                    'transcription': result
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/profiles', methods=['GET'])
        def get_profiles():
            """Get all profiles"""
            return jsonify({
                'success': True,
                'profiles': self.api.list_profiles()
            })
        
        @self.app.route('/profiles/<profile_name>', methods=['GET'])
        def get_profile(profile_name):
            """Get profile info"""
            try:
                profile_info = self.api.get_profile_info(profile_name)
                return jsonify({
                    'success': True,
                    'profile': profile_info
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 404
        
        @self.app.route('/profiles/emotions', methods=['GET'])
        def get_profile_emotions():
            """Get profiles grouped by emotion"""
            try:
                emotions = self.api.get_profile_emotions()
                return jsonify({
                    'success': True,
                    'emotions': emotions
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/profiles/emotions/<emotion>', methods=['GET'])
        def get_profiles_by_emotion(emotion):
            """Get profiles filtered by emotion"""
            try:
                profiles = self.api.get_profiles_by_emotion(emotion)
                return jsonify({
                    'success': True,
                    'emotion': emotion,
                    'profiles': profiles
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/profiles', methods=['POST'])
        def create_profile():
            """Create new profile"""
            try:
                # Handle both JSON and form-data
                if request.is_json:
                    data = request.get_json()
                    ref_audio_path = data['ref_audio_path']
                    temp_audio_path = None
                else:
                    # Handle form-data with file upload
                    data = request.form.to_dict()
                    audio_file = request.files.get('ref_audio')
                    
                    if audio_file:
                        temp_audio_path = self._handle_audio_file(audio_file.read(), f"profile_{data['profile_name']}")
                        ref_audio_path = temp_audio_path
                    elif 'ref_audio_path' in data:
                        ref_audio_path = data['ref_audio_path']
                        temp_audio_path = None
                    else:
                        return jsonify({'success': False, 'error': 'ref_audio file or ref_audio_path is required'}), 400
                
                success = self.api.create_profile(
                    profile_name=data['profile_name'],
                    ref_audio_path=ref_audio_path,
                    ref_text=data['ref_text'],
                    description=data.get('description', ''),
                    emotion=data.get('emotion', 'normal'),
                    overwrite=self._parse_bool(data.get('overwrite', False))
                )
                
                # Cleanup temporary file
                self._cleanup_temp_file(temp_audio_path)
                
                return jsonify({
                    'success': success,
                    'message': f"Profile '{data['profile_name']}' created successfully"
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/profiles/<profile_name>', methods=['DELETE'])
        def delete_profile(profile_name):
            """Delete profile"""
            try:
                success = self.api.delete_profile(profile_name)
                return jsonify({
                    'success': success,
                    'message': f"Profile '{profile_name}' deleted successfully"
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/info', methods=['GET'])
        def model_info():
            """Get model information"""
            return jsonify({
                'success': True,
                'info': self.api.get_model_info()
            })
    
    def run(self, debug=False):
        """Run the Flask server"""
        self.app.run(host=self.host, port=self.port, debug=debug)


# Command Line Interface and Examples
def run_example_1():
    """Example 1: Simple TTS"""
    print("=== Example 1: Simple TTS ===")
    
    # Initialize API
    api = F5TTSThaiAPI(model_type="Default")
    
    # Create output directory
    output_dir = "./api-out"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        result = api.infer_tts(
            ref_audio_path="./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",
            ref_text="ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น",
            gen_text="พรุ่งนี้มีประชุมสำคัญ อย่าลืมเตรียมเอกสารให้เรียบร้อย",
            output_path=os.path.join(output_dir, "output_simple.wav"),
            return_spectrogram=True
        )
        print(f"✓ Generated audio with sample rate: {result['sample_rate']}")
        print(f"✓ Audio shape: {result['audio_data'].shape}")
        print(f"✓ Seed used: {result['seed']}")
        print(f"✓ Output saved to: {os.path.join(output_dir, 'output_simple.wav')}")
        if 'spectrogram_path' in result:
            print(f"✓ Spectrogram saved to: {result['spectrogram_path']}")
    except Exception as e:
        print(f"✗ Error in simple TTS: {e}")


def run_example_2():
    """Example 2: Multistyle TTS"""
    print("=== Example 2: Multistyle TTS ===")
    
    # Initialize API
    api = F5TTSThaiAPI(model_type="Default")
    
    # Create output directory
    output_dir = "./api-out"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        speech_types = {
            "ปกติ": {
                "audio": "./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",
                "ref_text": "ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น"
            },
            "เศร้า": {
                "audio": "./src/f5_tts/infer/examples/thai_examples/ref_gen_2.wav",
                "ref_text": "ฉันเดินทางไปเที่ยวที่จังหวัดเชียงใหม่ในช่วงฤดูหนาวเพื่อสัมผัสอากาศเย็นสบาย"
            }
        }
        
        gen_text = "{ปกติ} สวัสดีครับ มีอะไรให้ผมช่วยไหมครับ {เศร้า} ผมเครียดจริงๆ นะตอนนี้"
        
        result = api.infer_multistyle(
            gen_text=gen_text,
            speech_types=speech_types,
            output_path=os.path.join(output_dir, "output_multistyle.wav"),
            return_spectrogram=True
        )
        print(f"✓ Generated multistyle audio with sample rate: {result['sample_rate']}")
        print(f"✓ Audio shape: {result['audio_data'].shape}")
        print(f"✓ Seed used: {result['seed']}")
        print(f"✓ Output saved to: {os.path.join(output_dir, 'output_multistyle.wav')}")
        if 'spectrogram_paths' in result and result['spectrogram_paths']:
            print(f"✓ Spectrograms saved to: {result['spectrogram_paths']}")
    except Exception as e:
        print(f"✗ Error in multistyle TTS: {e}")


def run_example_3():
    """Example 3: Transcription"""
    print("=== Example 3: Transcription ===")
    
    # Initialize API
    api = F5TTSThaiAPI(model_type="Default")
    
    try:
        transcription = api.transcribe_audio(
            audio_path="./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",
            translate=False,
            model="large-v3-turbo"
        )
        print(f"✓ Transcription: {transcription}")
    except Exception as e:
        print(f"✗ Error in transcription: {e}")


def run_example_4():
    """Example 4: Profile Management"""
    print("=== Example 4: Profile Management ===")
    
    # Initialize API
    api = F5TTSThaiAPI(model_type="Default")
    
    # Create output directory
    output_dir = "./api-out"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a profile
        profile_name = "thai_speaker_1"
        api.create_profile(
            profile_name=profile_name,
            ref_audio_path="./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",
            ref_text="ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น",
            description="Thai speaker with calm voice",
            emotion="normal",
            overwrite=True
        )
        
        # Create additional profiles with different emotions
        api.create_profile(
            profile_name="thai_speaker_happy",
            ref_audio_path="./src/f5_tts/infer/examples/thai_examples/ref_gen_2.wav",
            ref_text="ฉันเดินทางไปเที่ยวที่จังหวัดเชียงใหม่ในช่วงฤดูหนาวเพื่อสัมผัสอากาศเย็นสบาย",
            description="Thai speaker with happy voice",
            emotion="happy",
            overwrite=True
        )
        
        # Use profile for TTS
        result = api.infer_tts_with_profile(
            profile_name=profile_name,
            gen_text="วันนี้อากาศดีมาก เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ",
            output_path=os.path.join(output_dir, "output_with_profile.wav")
        )
        print(f"✓ Generated audio using profile '{profile_name}'")
        print(f"✓ Audio shape: {result['audio_data'].shape}")
        print(f"✓ Seed used: {result['seed']}")
        print(f"✓ Output saved to: {os.path.join(output_dir, 'output_with_profile.wav')}")
        
        # List available profiles
        profiles = api.list_profiles()
        print(f"✓ Available profiles: {profiles}")
        
        # Get profiles by emotion
        emotions = api.get_profile_emotions()
        print(f"✓ Profiles by emotion: {json.dumps(emotions, indent=2)}")
        
        # Test multistyle with profiles
        result = api.infer_multistyle_with_profiles(
            gen_text="{normal} สวัสดีครับ {happy} วันนี้ผมมีความสุขมาก",
            profile_emotions={
                "normal": "normal",  # Use emotion to find profile
                "happy": "happy"     # Use emotion to find profile
            },
            output_path=os.path.join(output_dir, "output_multistyle_profiles.wav")
        )
        print(f"✓ Generated multistyle audio using profile emotions")
        print(f"✓ Audio shape: {result['audio_data'].shape}")
        
        # Get profile info
        profile_info = api.get_profile_info(profile_name)
        print(f"✓ Profile info: {json.dumps(profile_info, indent=2)}")
        
    except Exception as e:
        print(f"✗ Error in profile example: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='F5-TTS Thai API')
    parser.add_argument('--example', type=str, choices=['1', '2', '3', '4'], 
                        help='Run specific example (1: Simple TTS, 2: Multistyle TTS, 3: Transcription, 4: Profile Management)')
    parser.add_argument('--server', action='store_true', help='Start REST API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=4000, help='Server port')
    parser.add_argument('--model', type=str, default='Default', choices=['Default', 'FP16', 'Custom'], 
                        help='Model type to use')
    parser.add_argument('--custom-model', type=str, help='Path to custom model (if model=Custom)')
    
    args = parser.parse_args()
    
    if args.example:
        # Run specific example
        if args.example == '1':
            run_example_1()
        elif args.example == '2':
            run_example_2()
        elif args.example == '3':
            run_example_3()
        elif args.example == '4':
            run_example_4()
    
    elif args.server:
        # Start REST API server
        print(f"Starting F5-TTS Thai API Server on {args.host}:{args.port}")
        api = F5TTSThaiAPI(model_type=args.model, custom_model_path=args.custom_model)
        
        if not FLASK_AVAILABLE:
            print("Error: Flask is not installed. Please install it with: pip install flask")
            return
        
        server = F5TTSAPIServer(api, host=args.host, port=args.port)
        server.run()
    
    else:
        # Show help if no arguments provided
        parser.print_help()


# Legacy example code (will be removed in future versions)
if __name__ == "__main__":
    main()
