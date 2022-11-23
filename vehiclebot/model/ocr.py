
from .model import HFTransformerModel

from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel
)

import numpy as np

class OCRModelTransformers(HFTransformerModel):
    @classmethod
    def _loadTransformer(cls, model_path : str) -> dict:
        net = {
            'processor': TrOCRProcessor.from_pretrained(model_path),
            'decoder': VisionEncoderDecoderModel.from_pretrained(model_path)
        }
        meta = {}
        
        return {
            "net" : net,
            "metadata" : meta
        }

    def detect(self, img : np.ndarray):
        #OCR pipeline
        pixel_values = self._net['processor'](img, return_tensors="pt").pixel_values 
        generated_ids = self._net['decoder'].generate(pixel_values, max_length=12)
        return self._net['processor'].batch_decode(generated_ids, skip_special_tokens=True)
            