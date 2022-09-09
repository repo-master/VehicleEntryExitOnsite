
from .model import Model

import cv2
import numpy as np
import numpy.typing as npt

class YOLOModel(Model):
    def detect(self,
                   img : npt.ArrayLike,
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

    def _phase_preprocess(self, img : npt.ArrayLike):
        return cv2.dnn.blobFromImage(img, 1 / 255.0, self._meta['net_input_size'],
            swapRB=True, crop=False)
    
    def _phase_forward(self, inp : npt.ArrayLike):
        self._net.setInput(inp)
        return self._net.forward(self._net.getUnconnectedOutLayersNames())
    
    def _phase_unwrap(self,
                      img_orig : npt.ArrayLike, detections,
                      min_confidence : float,
                      min_score : float,
                      label_str : bool) -> tuple:
        class_ids = []
        confidences = []
        boxes = []
        
        rows = detections[0].shape[1]
        image_height, image_width = img_orig.shape[:2]

        #Since we resized image to fit YOLO's layer shape, we need to rescale points to original
        x_factor = image_width / self._meta['net_input_size'][0]
        y_factor =  image_height / self._meta['net_input_size'][1]
        
        for r in range(rows):
            row = detections[0][0][r]
            confidence = row[4]
            if confidence >= min_confidence:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
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
        return np.array(boxes), np.array(confidences), np.array(class_ids)

    def _phase_nms(self, detections : tuple, min_confidence : float, min_nms : float) -> tuple:
        boxes, confidences, class_ids = detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, min_nms)
        return boxes.take(indices, axis=0), confidences.take(indices, axis=0), class_ids.take(indices, axis=0)

    def _phase_zip(self, detections : tuple) -> list:
        return list(zip(*detections))
