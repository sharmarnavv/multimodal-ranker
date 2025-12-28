import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import laion_clap
import librosa
import numpy as np
import os

class VibeEngine:
    def __init__(self):
        # 1. Hardware Detection
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"🚀 Loading Multimodal Models on {self.device}...")

        # 2. Load Visual Model (CLIP)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 3. Load Audio Model (CLAP)
        # enable_fusion=False is standard for simple embedding generation
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt() # Downloads model (~1.5GB) on first run
        self.clap_model.to(self.device)
        
        print("Vibe Engine (Visual + Audio) Ready.")

    def get_image_vector(self, image_path):
        """Image -> 512d CLIP Vector"""
        if not image_path or not os.path.exists(image_path):
            return None
            
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            vector = self.clip_model.get_image_features(**inputs)
        
        # Normalize
        norm_vector = vector / vector.norm(p=2, dim=-1, keepdim=True)
        return norm_vector.cpu().numpy()[0].tolist()

    def get_audio_vector(self, audio_path):
        """Audio File -> 512d CLAP Vector"""
        if not audio_path or not os.path.exists(audio_path):
            return None

        # Load and Resample to 48kHz (Required by CLAP)
        try:
            audio_data, _ = librosa.load(audio_path, sr=48000)
            audio_data = audio_data.reshape(1, -1) # Add batch dim

            with torch.no_grad():
                vector = self.clap_model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
            
            # CLAP output is usually (1, 512)
            return vector[0].tolist()
        except Exception as e:
            print(f"⚠️ Audio Processing Error: {e}")
            return None

    def get_text_vector(self, text, modality="visual"):
        """
        modality: 
          - 'visual': Maps text to CLIP space (for searching images)
          - 'audio': Maps text to CLAP space (for searching audio)
        """
        if modality == "visual":
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                vector = self.clip_model.get_text_features(**inputs)
            norm_vector = vector / vector.norm(p=2, dim=-1, keepdim=True)
            return norm_vector.cpu().numpy()[0].tolist()
            
        elif modality == "audio":
            # CLAP handles its own text tokenization
            vector = self.clap_model.get_text_embedding([text])
            return vector[0].tolist()