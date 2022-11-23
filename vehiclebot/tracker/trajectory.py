
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
        self.tracks : typing.Dict[int, typing.Dict] = {}

    def __getitem__(self, track_id : int):
        return self.tracks[track_id]

    def update(self, track_id, centroid_point):
        if track_id not in self.tracks:
            self.tracks[track_id] = {"pts": [], 'dir': 0.0, 'cardinal': '', 'bz': [], 'speed': 0.0, 'is_moving': False, 'count_moving': 0, 'active': True}
        self.tracks[track_id]["pts"].append(centroid_point)

        if len(self.tracks[track_id]['pts']) >= 2:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    bz_curve = bezier.bezier_curve(self.tracks[track_id]['pts'][-15:], nTimes=20, t_start=0.0, t_end=0.33)
                    bz_coords = list(zip(*bz_curve))
                    bz_p1, bz_p2 = bz_coords[0], bz_coords[-1]
                    
                    #slope = np.polyfit(*bz_curve, 1)[0]
                    dx = bz_p1[0]-bz_p2[0]
                    dy = bz_p1[1]-bz_p2[1]
                    angle = np.arctan2(dy, dx)
                    speed = np.sqrt(np.square(dy)+np.square(dx))
                    is_moving = speed >= 15
                    
                    if is_moving:
                        self.tracks[track_id]['count_moving'] += 1
                        if self.tracks[track_id]['count_moving'] > 2:
                            self.tracks[track_id]['count_moving'] = 2
                            self.tracks[track_id]['is_moving'] = True
                    else:
                        self.tracks[track_id]['count_moving'] -= 1
                        if self.tracks[track_id]['count_moving'] < 0:
                            self.tracks[track_id]['count_moving'] = 0
                            self.tracks[track_id]['is_moving'] = False

                    self.tracks[track_id]['bz'] = bz_curve
                    self.tracks[track_id]['dir'] = angle
                    self.tracks[track_id]['cardinal'] = self.degToCompass(self.rad2deg(angle)) if is_moving else 'Stopped'
                    self.tracks[track_id]['speed'] = speed
                except np.RankWarning:
                    pass
                
    @staticmethod
    def rad2deg(r):
        return (np.rad2deg(np.pi/2+r) + 360) % 360

    @staticmethod
    def degToCompass(num):
        '''
        Convert angle to compass direction
        Taken from https://stackoverflow.com/a/7490772
        '''
        val = int((num/22.5)+.5)
        arr = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
        return arr[(val % 16)]

    @staticmethod
    def tupleToInt(x):
        return tuple(map(int, x))

    def draw(self, img : npt.ArrayLike):
        for trk_id, track in self.tracks.items():
            if len(track['pts']) >= 2 and track['active']:
                cv2.polylines(img, [np.array([self.tupleToInt(x) for x in zip(*track['bz'])])], False, (255,0,0), 2)
                #for px, py in zip(*track['bz']):
                #    cv2.circle(img, (int(px), int(py)), 2, (255,255,255), -1)

                last_pt = track['pts'][-1]
                ca = np.cos(track['dir'])*30
                sa = np.sin(track['dir'])*30

                p1x = last_pt[0]-ca
                p1y = last_pt[1]-sa
                p2x = last_pt[0]+ca
                p2y = last_pt[1]+sa
                
                #cv2.arrowedLine(img, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0,0,0), thickness=3)
                #cv2.arrowedLine(img, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (255,255,255), thickness=2)
                #Track direction
                dir_text = "Dir: %s (speed %.2f)" % (track['cardinal'], track['speed'])
                cv2.putText(img, dir_text, (int(last_pt[0]) - 10, int(last_pt[1]) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
                cv2.putText(img, dir_text, (int(last_pt[0]) - 10, int(last_pt[1]) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), lineType=cv2.LINE_AA, thickness=1)
            
