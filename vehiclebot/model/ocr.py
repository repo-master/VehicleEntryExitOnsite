
from .model import HFTransformerModel, TorchModel

from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel
)

import numpy as np
import torch

class OCRModelTransformers(HFTransformerModel):
    @classmethod
    def _loadTransformer(cls, model_path : str, device : str = None, cache_dir = None) -> dict:
        meta = {}
        device = TorchModel.getDevice(device)
        meta['device'] = device

        net = {
            'processor': TrOCRProcessor.from_pretrained(model_path, cache_dir=cache_dir),
            'decoder': VisionEncoderDecoderModel.from_pretrained(model_path, cache_dir=cache_dir).to(device)
        }
        
        return {
            "net" : net,
            "metadata" : meta
        }

    def detect(self, img : np.ndarray, max_length=12):
        device = self.metadata['device']

        #OCR pipeline
        with torch.no_grad():
            pixel_values = self.net['processor'](img, return_tensors="pt").pixel_values
            generated_ids = self.net['decoder'].generate(pixel_values.to(device), max_length=max_length)
            return self.net['processor'].batch_decode(generated_ids, skip_special_tokens=True)
        