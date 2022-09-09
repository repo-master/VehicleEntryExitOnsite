
import typing

import cv2
import motrackers.track
import numpy as np
import numpy.typing as npt
import warnings

from . import bezier

Detections = typing.OrderedDict[int, motrackers.track.Track]

TRACK_COLORS = [
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (0,255,255),
    (255,0,255),
    (128,64,255),
    (64,128,255),
    
    #Unknown/new/lost track
    (192,192,192)
]

class Trajectory:
    def __init__(self):
        self.tracks = {}

    def update(self, track_id, centroid_point, is_active : bool = True):
        if track_id not in self.tracks:
            self.tracks[track_id] = {"pts": [], 'dir': 0.0, 'bz': []}
        self.tracks[track_id]["pts"].append(centroid_point)

        if len(self.tracks[track_id]['pts']) >= 2:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    bz_curve = bezier.bezier_curve(self.tracks[track_id]['pts'][-30:], nTimes=90, t_start=0, t_end=1)
                    slope = np.polyfit(*bz_curve, 1)[0]
                    self.tracks[track_id]['bz'] = bz_curve
                    self.tracks[track_id]['dir'] = slope
                except np.RankWarning:
                    pass

    def estimatedDirection(self, track_id):
        pass

    @staticmethod
    def tupleToInt(x):
        return tuple(map(int, x))

    def draw(self, img : npt.ArrayLike):
        for trk_id, track in self.tracks.items():
            if len(track['pts']) >= 2:
                cv2.polylines(img, [np.array([self.tupleToInt(x) for x in zip(*track['bz'])])], False, (255,255,255), 2)
                #for px, py in zip(*track['bz']):
                #    cv2.circle(img, (int(px), int(py)), 2, (255,255,255), -1)

                last_pt = track['pts'][-1]
                ca = np.cos(track['dir'])*30
                sa = np.sin(track['dir'])*30

                p1x = last_pt[0]-ca
                p1y = last_pt[1]-sa
                p2x = last_pt[0]+ca
                p2y = last_pt[1]+sa
                
                cv2.arrowedLine(img, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0,0,0), thickness=3)
                cv2.arrowedLine(img, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (255,255,255), thickness=2)
