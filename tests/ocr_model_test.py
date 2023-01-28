
import unittest
from parameterized import parameterized
from contextlib import suppress

import pandas as pd
from fuzzywuzzy import fuzz

import os
import re

#
import cv2
import numpy as np
from vehiclebot.model.ocr import OCRModelTransformers
from vehiclebot.imutils import scaleImgRes
import pytesseract

TEST_PATH = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(TEST_PATH, "data/")

HF_MODEL_CACHEDIR = "models/.transformer-cache/"

#CrossML dataset
CROSSML_DF = pd.read_csv(os.path.join(TEST_DATA_PATH, "crossml/labels.csv"))
CROSSML_DF['File'] = os.path.join(TEST_DATA_PATH, "crossml/") + CROSSML_DF['File']
CROSSML_DF.dropna(inplace=True, axis=0)
CROSSML_DF = CROSSML_DF[CROSSML_DF['label'].str.contains('\*')==False]

#A dataset from Kaggle. Only testing images are marked
KAGGLE_TEST_DF = pd.read_csv(os.path.join(TEST_DATA_PATH, "kaggle/testdb.csv"))

TROCR_MODELS_TO_TEST = [
    ('microsoft/trocr-small-printed', 'microsoft/trocr-small-printed'),
    ('models/anpr/anpr_demo/', 'microsoft/trocr-small-printed'),
    ('models/anpr/anpr_20230124T064839/', 'microsoft/trocr-small-printed'),
    ('models/anpr/anpr_20230125T135152/', 'microsoft/trocr-small-printed'),
    ('models/anpr/anpr_20230125T171217/', 'microsoft/trocr-small-printed'),
    ('models/anpr/anpr_20230126T160350/', 'microsoft/trocr-small-printed'),
    ('models/anpr/anpr_20230126T180659/', 'microsoft/trocr-small-printed'),
]

class TestOCROutputCrossMLDataset(unittest.TestCase):
    DATASET = CROSSML_DF
    ALL_MODELS = TROCR_MODELS_TO_TEST
    MIN_MATCH_PERCENT = 90

    def iterdataset(self):
        for idx, row in self.DATASET.iterrows():
            img_path, ground_truth = row

            ITEM_MSG = "Load Image idx {idx}".format(idx=idx)
            with self.subTest(msg=ITEM_MSG):
                img = cv2.imread(img_path)

                # Make sure image is read correctly and is a 3-channel image
                self.assertIsNotNone(img)
                self.assertEqual(len(img.shape), 3)
                self.assertEqual(img.shape[2], 3)

                # Preprocessing
                img, _ = scaleImgRes(img, height=32)
                # Make image grayscale, then to bgr again, effectively stripping colors
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                yield (
                    idx,
                    img,
                    img_path,
                    ground_truth
                )

    @parameterized.expand(ALL_MODELS)
    def test_trocr_output_image(self, trocr_visioncodec_model : os.PathLike, trocr_processor : os.PathLike = None):
        model = OCRModelTransformers.fromHuggingFace(model_path=trocr_visioncodec_model, processor_path=trocr_processor, cache_dir=HF_MODEL_CACHEDIR)

        data = self.iterdataset()

        for idx, img, img_path, ground_truth in data:
            ITEM_MSG = "{idx}_{gt}".format(idx=idx, gt=ground_truth)
            with self.subTest(msg=ITEM_MSG):
                detected_text = model.detect(img)

                text_combined = ' '.join(detected_text)
                text_combined = re.sub('[^\w\s]+', '', text_combined)

                match_ratio = fuzz.ratio(ground_truth, text_combined)

                if match_ratio < self.MIN_MATCH_PERCENT:
                    self.assertEqual(text_combined, ground_truth)

    def test_tesseract_output_image(self):
        data = self.iterdataset()

        for idx, img, img_path, ground_truth in data:
            ITEM_MSG = "{idx}_{gt}".format(idx=idx, gt=ground_truth)
            with self.subTest(msg=ITEM_MSG):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                detected_text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')

                text_combined = ' '.join(detected_text)
                text_combined = re.sub('[^\w\s]+', '', text_combined)

                match_ratio = fuzz.ratio(ground_truth, text_combined)

                if match_ratio < self.MIN_MATCH_PERCENT:
                    self.assertEqual(text_combined, ground_truth)

    @parameterized.expand([
        [(10,1)],
        [(1,10)],
        [(10,10)],
        [(25,25)],
        [(25,1)],
        [(1,25)]
    ])
    def test_tesseract_grid_output_image(self, grid_size=(1,1)):
        data = self.iterdataset()

        for idx, img, img_path, ground_truth in data:
            ITEM_MSG = "{idx}_{gt}".format(idx=idx, gt=ground_truth)
            with self.subTest(msg=ITEM_MSG):
                new_size = (img.shape[0]*grid_size[1], img.shape[1]*grid_size[0], img.shape[2])
                img_grid = np.zeros(new_size, dtype=img.dtype)
                for x in range(grid_size[0]):
                    x *= img.shape[1]
                    for y in range(grid_size[1]):
                        y *= img.shape[0]
                        with suppress(ValueError):
                            img_grid[y:y+img.shape[0],x:x+img.shape[1]] = img

                gray = cv2.cvtColor(img_grid, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                detected_text = pytesseract.image_to_string(thresh)

                text_combined = ' '.join(detected_text)
                text_combined = re.sub('[^\w\s]+', '', text_combined)

                match_ratio = fuzz.ratio(ground_truth, text_combined)

                if match_ratio < self.MIN_MATCH_PERCENT:
                    self.assertEqual(text_combined, ground_truth)

import HtmlTestRunner
if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_results/'))
