
from vehiclebot.task import AIOTask
from vehiclebot.multitask import AsyncProcess
from vehiclebot.model import YOLOModelTransformers, OCRModelTransformers
from vehiclebot.model.augmented_filter import augmentation

import typing
from datetime import datetime
import cv2
import numpy as np
import re

#Use huggingface transformers model to detect license plate
#and an OCR to decode it

NUMBER_PLATE_PATTERN = re.compile(
    r"^(?P<stateCode>[a-z]{2})\s?(?P<rtoCode>\d{1,2})\s?(?P<series>[a-z]{0,2})\s?(?P<vehicleCode>\d{4})$",
    re.MULTILINE | re.IGNORECASE)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

class LicensePlateProcess:
    def __init__(self):
        self.__detect_model_file = "nickmuchi/yolos-small-finetuned-license-plate-detection"
        self.__decode_model_file = "microsoft/trocr-small-printed"
        
        #Load model
        self.model = {
            'detect': YOLOModelTransformers.fromHuggingFace(self.__detect_model_file),
            'decode': OCRModelTransformers.fromHuggingFace(self.__decode_model_file)
        }

    def detectAndDecode(self, detection : dict):
        #Image License plate recognition pipeline
        plate_name_candidates = []
        
        #Stage 0: Re-decode the image (vehicle, cropped)
        img_blob = detection['img']['data']
        img = cv2.imdecode(img_blob, cv2.IMREAD_COLOR)
        
        #Stage 1: Detect license plate(s?!!)
        res = self.model['detect'].detect(img)

        #Iterate over the detections
        for i, (box, conf, clsid) in enumerate(res):
            #Crop just the plate (with some padding)
            padding = 0
            pt1 = (int(box[0]-padding), int(box[1]-padding))
            pt2 = (int(box[2]+padding), int(box[3]+padding))
            img_cropped = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
            
            #if image detection was incorrect
            if img_cropped.shape[0]*img_cropped.shape[1] == 0:
                continue
            
            #Resize to a standard resolution
            img_aspect = img_cropped.shape[1]/img_cropped.shape[0]
            ##Specify Width. height is automatic
            new_width = 512 #Can change this
            new_height = int(new_width/img_aspect)

            #GRAYSCALE, Resize to the new image size
            img_raw = cv2.cvtColor(
                cv2.resize(img_cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_BGR2GRAY
            )

            MAX_AUGS = 12
            imgs_aug = augmentation(img_raw, num_iters = MAX_AUGS, aug_max_combinations=0)
            #BGR for text detection
            imgs_aug = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in imgs_aug]
            
            #Stage 2: OCR
            generated_text = self.model['decode'].detect(imgs_aug)
            
            all_plate_matches = [
                #Remove non-alphanumeric characters, and validate against the license plate pattern
                NUMBER_PLATE_PATTERN.match(re.sub("\W+", "", x))
                for x in generated_text
            ]
            all_plate_matches = [x.groupdict() if x else {} for x in all_plate_matches]
            #Histogram
            hist = {}
            for i in all_plate_matches:
                for j,k in i.items():
                    if j not in hist:
                        hist[j] = {}
                    if len(k) == 0: continue
                    if k not in hist[j]:
                        hist[j][k] = 1
                    else:
                        hist[j][k] += 1
                        
            best_with_parts = {k : max(v.items(), key=lambda x: x[1]) for k,v in hist.items()}
            if len(best_with_parts) > 0:
                best_acc = (
                    ' '.join([x[0] for x in best_with_parts.values()]),
                    sum([x[1] for x in best_with_parts.values()])/MAX_AUGS
                )
                plate_name_candidates.append(best_acc)
            
        if len(plate_name_candidates) > 0:
            bst_plt = plate_name_candidates[0]
            return {
                "plate_number": bst_plt[0],
                "accuracy": bst_plt[1],
                "detect_ts": datetime.now()
            }
    

class PlateRecogniser(AIOTask, AsyncProcess):
    metadata : typing.Dict[str, typing.Any] = {"dependencies": []}
    def __init__(self, tm, task_name, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.plate_decoder : LicensePlateProcess = None
        self.prepareProcess()
        self.logger.info("Started License plate worker process")
        
    async def start_task(self):
        try:
            self.logger.info("Loading model for license plate...")
            self.plate_decoder = await self.asyncCreate(LicensePlateProcess)
        except (FileNotFoundError, AttributeError, ValueError):
            self.logger.exception("Error creating License Plate detector")
            
    async def stop_task(self):
        await self.task
        self.endProcess()
        
    async def detectAndDecode(self, det : dict):
        if not self.plate_decoder: return
        return await self.plate_decoder.detectAndDecode(det)
