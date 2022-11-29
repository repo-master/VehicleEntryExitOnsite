
import typing

import cv2
import numpy as np
import warnings

import motrackers.track
from motrackers.tracker import Tracker

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
    def __init__(self, tracker : Tracker):
        self._tracker = tracker
        self._max_bezier_lookback_size = 15
        self._min_movement_detection_speed = 2.0
        self._track_smooth_coefficient = 0.2
    
    def trackerID(self):
        return "traj_%d" % id(self)
    
    def update(self, track_centroid_source : typing.Callable[[motrackers.track.Track],typing.Tuple] = None):
        _tracker_id = self.trackerID()
        for trk_obj in self._tracker.tracks.values():
            if not hasattr(trk_obj, _tracker_id):
                setattr(trk_obj, _tracker_id, {
                    'pts': [],
                    'curr_pos': None,
                    'dir': 0.0,
                    'cardinal': '',
                    'bz': [],
                    'speed': 0.0,
                    'is_moving': False,
                    'count_moving': 0,
                    'active': True
                })
            traj_data = getattr(trk_obj, _tracker_id)
            
            trk_bbox = None
            if track_centroid_source is not None:
                trk_bbox = track_centroid_source(trk_obj)

            if trk_bbox is None:
                trk_bbox = trk_obj.bbox

            xmin, ymin, width, height = trk_bbox
            centroid = (xmin + 0.5*width, ymin + 0.5*height)
            
            if traj_data['curr_pos'] is None:
                traj_data['curr_pos'] = centroid
            else:
                curr_pos = traj_data['curr_pos']
                centroid_smooth = (
                    traj_data['curr_pos'][0] + (centroid[0] - curr_pos[0]) * self._track_smooth_coefficient,
                    traj_data['curr_pos'][1] + (centroid[1] - curr_pos[1]) * self._track_smooth_coefficient,
                )
                traj_data['curr_pos'] = centroid_smooth

            traj_data['pts'].append(traj_data['curr_pos'])

            #At least two points needed to form a bezier curve
            if len(traj_data['pts']) >= 2:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        #Generate bezier curve
                        bz_curve = bezier.bezier_curve(traj_data['pts'][-self._max_bezier_lookback_size:], nTimes=20, t_start=0.0, t_end=0.33)
                        bz_coords = list(zip(*bz_curve))
                        bz_p1, bz_p2 = bz_coords[0], bz_coords[-1]
                        
                        dx = bz_p1[0]-bz_p2[0]
                        dy = bz_p1[1]-bz_p2[1]
                        angle = np.arctan2(dy, dx)
                        speed = np.sqrt(np.square(dy)+np.square(dx))
                        is_moving = speed >= self._min_movement_detection_speed
                    
                        if is_moving:
                            traj_data['count_moving'] += 1
                            if traj_data['count_moving'] > 3:
                                traj_data['count_moving'] = 3
                                traj_data['is_moving'] = True
                        else:
                            traj_data['count_moving'] -= 1
                            if traj_data['count_moving'] < 0:
                                traj_data['count_moving'] = 0
                                traj_data['is_moving'] = False

                        traj_data['bz'] = bz_coords
                        traj_data['dir'] = angle
                        traj_data['cardinal'] = self.degToCompass(self.rad2deg(angle)) if is_moving else 'Stopped'
                        traj_data['speed'] = speed
                    except np.RankWarning:
                        pass
                
    __call__ = update

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
    
    def draw(self, trk_obj, img : np.ndarray, scale : float = 1.0):
        _tracker_id = self.trackerID()
        if hasattr(trk_obj, _tracker_id):
            traj_data = getattr(trk_obj, _tracker_id)
            if len(traj_data['pts']) >= 2 and traj_data['active']:
                cv2.polylines(img, [np.array([(int(x*scale),int(y*scale)) for x,y in traj_data['bz']])], False, (255,0,0), 3)
                #for px, py in traj_data['bz']:
                #    cv2.circle(img, (int(px*scale), int(py*scale)), 2, (255,255,255), -1)

                last_pt = traj_data['pts'][-1]
                last_pt = (last_pt[0]*scale, last_pt[1]*scale)
                #ca = np.cos(traj_data['dir'])*30
                #sa = np.sin(traj_data['dir'])*30
                #
                #p1x = last_pt[0]-ca
                #p1y = last_pt[1]-sa
                #p2x = last_pt[0]+ca
                #p2y = last_pt[1]+sa
                
                #cv2.arrowedLine(img, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0,0,0), thickness=3)
                #cv2.arrowedLine(img, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (255,255,255), thickness=2)
                #Track direction
                dir_text = "Dir: %s (speed %.2f)" % (traj_data['cardinal'], traj_data['speed'])
                cv2.putText(img, dir_text, (int(last_pt[0]) - 10, int(last_pt[1]) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
                cv2.putText(img, dir_text, (int(last_pt[0]) - 10, int(last_pt[1]) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), lineType=cv2.LINE_AA, thickness=1)
            
