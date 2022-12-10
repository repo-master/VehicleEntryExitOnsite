
import os
import json
import typing
import zipfile

import cv2
import numpy as np

class Model:
    def __init__(self, net, class_labels : typing.List[str] = None, metadata : typing.Dict = dict()):
        self._net = net
        self._meta = metadata
        self.class_labels = class_labels
    
    def detect(self, img : np.ndarray, *args, **kwargs):
        pass

    def getLabel(self, idx : int) -> str:
        if not self.class_labels: return
        return self.class_labels[idx]
    
class CV2ModelZipped(Model):
    @classmethod
    def fromZip(cls, model_zip_file : os.PathLike):
        with zipfile.ZipFile(model_zip_file, 'r') as zf:
            return cls(**cls._loadNetZip(zf))

    @classmethod
    def _loadNetZip(cls, zf : zipfile.ZipFile):
        meta = json.load(zf.open("meta.json"))
        labels = []
        
        #Load network structure and weights
        with zf.open(meta['net_file'], 'r') as netf:
            net = cls.loadNetFromBuffer(netf.read(), meta['framework'])
        
        if isinstance(meta['net_labels'], list):
            labels = meta['net_labels']
        elif isinstance(meta['net_labels'], str):
            with zf.open(meta['net_labels'], 'r') as lblf:
                labels = [x.strip().decode() for x in lblf.readlines() if len(x.strip())>0]
                
        return {
            "net" : net,
            "class_labels": labels,
            "metadata" : meta
        }

    @staticmethod
    def loadNetFromBuffer(buf : bytes, framework : str = "onnx") -> cv2.dnn.Net:
        '''
        Create an OpenCV Net object from the model data provided and the framework to use.
        Currently OpenCV does not support loading from buffer using 'cv2.readNet' for some types,
        so this is a replacement for that.
        '''
        fw_net_map = {
            'onnx': cv2.dnn.readNetFromONNX
        }
        try:
            net_fn = fw_net_map[framework]
            return net_fn(buf)
        except KeyError:
            raise ValueError("The provided framework '%s' is not supported" % framework)

class HFTransformerModel(Model):
    @classmethod
    def fromHuggingFace(cls, model_path : str, *args, **kwargs):
        return cls(**cls._loadTransformer(model_path, *args, **kwargs))

    #Depends on type of model
    @classmethod
    def _loadTransformer(cls, model_path : str, **kwargs) -> dict:
        raise NotImplementedError()
