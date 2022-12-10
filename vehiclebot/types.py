
import typing
import numpy as np
import re

class PlateDetection(dict):
    pass

Detections = typing.NamedTuple(
    "Detections",
    bboxes = np.ndarray,
    detection_scores = np.ndarray,
    class_ids = np.ndarray
)

PLATE_PATTERNS = {
    'in': r"(?P<stateCode>[a-z]{2})\s?(?P<rtoCode>\d{1,2})\s?(?P<series>\w{0,2})\s?(?P<vehicleCode>\d{4})"
}

NUMBER_PLATE_PATTERN = re.compile(PLATE_PATTERNS['in'], re.MULTILINE | re.IGNORECASE)
