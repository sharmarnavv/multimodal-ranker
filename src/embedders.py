import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

class VisualEngine:
    def __init__(self):
        # auto-detect hardware
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"🚀 Loading CLIP on {self.device}...")

        # load clip (vit-base)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("loaded vision engine")

    def get_image_vector(self, image_path):
        """image -> 512d vector"""
        if not image_path or not os.path.exists(image_path):
            return None
            
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            vector = self.model.get_image_features(**inputs)
        
        # normalize vector (needed for cosine sim)
        norm_vector = vector / vector.norm(p=2, dim=-1, keepdim=True)
        return norm_vector.cpu().numpy()[0].tolist()

    def get_text_vector(self, text):
        """search query -> 512d vector"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            vector = self.model.get_text_features(**inputs)
            
        norm_vector = vector / vector.norm(p=2, dim=-1, keepdim=True)
        return norm_vector.cpu().numpy()[0].tolist()