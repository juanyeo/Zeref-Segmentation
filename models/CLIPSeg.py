import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class CLIPSeg:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    def predict(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = torch.sigmoid(outputs.logits) # shape: 352 x 352

        return prediction