
from .model import HFTransformerModel

from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel
)

import numpy as np
import torch

class OCRModelTransformers(HFTransformerModel):
    @classmethod
    def _loadTransformer(cls, model_path : str, device : str = None, cache_dir = None) -> dict:
        meta = {}
        
        if device is None:
            #Can use NVIDIA GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        meta['device'] = device

        net = {
            'processor': TrOCRProcessor.from_pretrained(model_path, cache_dir=cache_dir),
            'decoder': VisionEncoderDecoderModel.from_pretrained(model_path, cache_dir=cache_dir).to(device)
        }
        
        return {
            "net" : net,
            "metadata" : meta
        }

    def detect(self, img : np.ndarray):
        device = self._meta['device']

        #OCR pipeline
        pixel_values = self._net['processor'](img, return_tensors="pt").pixel_values
        generated_ids = self._net['decoder'].generate(pixel_values.to(device), max_length=12)
        return self._net['processor'].batch_decode(generated_ids, skip_special_tokens=True)
        