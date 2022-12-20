
from .model import Model, TorchModel
from vehiclebot import imutils

import numpy as np
import typing

import cv2

from craft import CRAFT
import craft.craft_utils

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class CRAFTModel(Model):
    @classmethod
    def fromCRAFT(cls, model_path : str, **kwargs):
        return cls(**cls._loadCRAFT(model_path, **kwargs))

    @classmethod
    def _loadCRAFT(cls, model_path : str, device : str = None) -> dict:
        meta = {}
        device = TorchModel.getDevice(device)
        meta['device'] = device

        net = CRAFT()
        
        #Load pre-trained model and weights
        net.load_state_dict(copyStateDict(torch.load(model_path, map_location=device)))

        if device.type != 'cpu':
            #GPU (CUDA) mode with one or more GPUs
            net = torch.nn.DataParallel(net).to(device)
            cudnn.benchmark = False

        #Evaluation mode
        net.eval()

        return {
            "net" : net,
            "metadata" : meta
        }


    def detect(self,
        img : np.ndarray,
        min_confidence : float = 0.55,
        min_score : float = 0.6,
        min_nms : float = 0.45):
        
        device = self.metadata['device']

        #Invalid image
        if img.shape[0]*img.shape[1] == 0: return
        
        img_resized, target_ratio, size_heatmap = imutils.resize_aspect_ratio(img/255, 1280, cv2.INTER_CUBIC)
        ratio_h = ratio_w = 1 / target_ratio
        
        x = torch.from_numpy(img_resized).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        x = x.to(device)
        
        with torch.no_grad():
            y, _ = self.net(x)
        
        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # Post-processing
        boxes, _ = craft.craft_utils.getDetBoxes(score_text, score_link, 0.3, 0.2, 0.4, False)

        # coordinate adjustment
        boxes = craft.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        return boxes
