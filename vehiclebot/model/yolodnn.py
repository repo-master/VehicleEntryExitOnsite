
from .model import Model, CV2ModelZipped, HFTransformerModel

import numpy as np
import typing

import cv2
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import torch

class YOLOModel(Model):
    _NEED_UNSCALE = False
    def detect(self,
        img : np.ndarray,
        zip_results : bool = True,
        label_str : bool = False,
        min_confidence : float = 0.55,
        min_score : float = 0.6,
        min_nms : float = 0.45):
        
        blob = self._phase_preprocess(img)
        outputs = self._phase_forward(blob)
        detections = self._phase_unwrap(img, outputs, min_confidence, min_score, label_str)
        detections = self._phase_nms(detections, min_confidence, min_nms)
        if zip_results: detections = self._phase_zip(detections)
        return detections

    def _phase_preprocess(self, img : np.ndarray):
        raise NotImplementedError()
    
    def _phase_forward(self, inp):
        raise NotImplementedError()
    
    def _phase_unwrap(self,
                      img_orig : np.ndarray,
                      detections,
                      min_confidence : float,
                      min_score : float,
                      label_str : bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_ids = []
        confidences = []
        boxes = []
        
        rows = detections[0].shape[1]
        image_height, image_width = img_orig.shape[:2]
        
        #Since we resized image to fit YOLO's layer shape, we need to rescale points to original
        if self._NEED_UNSCALE:
            net_size = self._meta['net_input_size']
            x_factor = image_width / net_size[0]
            y_factor =  image_height / net_size[1]
        else:
            x_factor = 1
            y_factor = 1

        for r in range(rows):
            row = detections[0][0][r]
            confidence = row[4]
            if confidence >= min_confidence:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                #Cull low score results
                if classes_scores[class_id] > min_score:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])

                    boxes.append(box)
        
        if label_str:
            class_ids = list(map(self.getLabel, class_ids))

        return (np.array(boxes), np.array(confidences), np.array(class_ids))

    def _phase_nms(self, detections : tuple, min_confidence : float, min_nms : float) -> tuple:
        boxes, confidences, class_ids = detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, min_nms)
        return boxes.take(indices, axis=0), confidences.take(indices, axis=0), class_ids.take(indices, axis=0)

    def _phase_zip(self, detections : tuple) -> list:
        return list(zip(*detections))

class YOLOModelCV2(CV2ModelZipped, YOLOModel):
    _NEED_UNSCALE = True
    def _phase_preprocess(self, img : np.ndarray):
        frame_size = self._meta['net_input_size']

        cropped_img = img
        
        real_h, real_w = img.shape[:2]
        dnn_w, dnn_h = frame_size
        scale_x = real_w/dnn_w
        scale_y = real_h/dnn_h

        new_wh1 = (real_w, dnn_h*scale_x)
        new_wh2 = (dnn_w*scale_y, real_h)

        bestfit_wh = (int(min(new_wh1[0], new_wh2[0])), int(min(new_wh1[1], new_wh2[1])))
        gapx, gapy = real_w-bestfit_wh[0], real_h-bestfit_wh[1]
        
        if gapy > 0:
            cy1 = gapy//2
            cy2 = cropped_img.shape[0]-cy1
            cropped_img = cropped_img[cy1:cy2,:,:]
            
        return cv2.dnn.blobFromImage(img, 1 / 255.0, frame_size,
            swapRB=True, crop=False)
    
    def _phase_forward(self, inp : np.ndarray):
        self._net.setInput(inp)
        return self._net.forward(self._net.getUnconnectedOutLayersNames())
    
class YOLOModelTorch(YOLOModel):
    pass

class YOLOModelTransformers(HFTransformerModel, YOLOModel):
    '''
    Model to detect objects using the HuggingFace Transformers API
    '''
    @classmethod
    def _loadTransformer(cls, model_path : str, device : str = None, cache_dir = None) -> dict:
        labels = []
        meta = {}
        
        if device is None:
            #Can use NVIDIA GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        meta['device'] = device
        
        net = {
            'exractor': AutoFeatureExtractor.from_pretrained(model_path, cache_dir=cache_dir),
            'detector': AutoModelForObjectDetection.from_pretrained(model_path, cache_dir=cache_dir).to(device)
        }
        
        return {
            "net" : net,
            "class_labels": labels,
            "metadata" : meta
        }

    def _phase_preprocess(self, img : np.ndarray) -> dict:
        device = self._meta['device']
        feature_tensors = self._net['exractor'](images=[img], return_tensors="pt").pixel_values
        return feature_tensors.to(device)
        
    def _phase_forward(self, inp):
        return self._net['detector'](pixel_values = inp)
        
    def _phase_unwrap(self,
                      img_orig : np.ndarray,
                      detections,
                      min_confidence : float,
                      min_score : float,
                      label_str : bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_ids = []
        confidences = []
        boxes = []
        
        device = self._meta['device']
        img_sizes = torch.Tensor([img_orig.shape[:2]]).to(device)
        results = self._net['exractor'].post_process_object_detection(detections, threshold=min_confidence, target_sizes=img_sizes)
        #0th because one image, but loop in all detections
        result = results[0]
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            #Cull low score results
            if score > min_score:
                confidences.append(score.detach().cpu().numpy())
                boxes.append(box.detach().cpu().numpy())
                class_ids.append(label.detach().cpu().numpy())
        
        return (
            np.array(boxes),
            np.array(confidences),
            np.array(class_ids)
        )
