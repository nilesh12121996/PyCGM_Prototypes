import time
from math import *
import functools
import numpy as np

def parameters(*args):
    # decorator parameters are keys
    # finds key in either the marker data or the measurement data
    # sets function parameters accordingly
    def decorator(func):
        params = list(args)

        @functools.wraps(func)
        def set_required_markers(self, *args):
            required_params = [self.find(param) for param in params]
            return func(self, *required_params)
        return set_required_markers
    return decorator


class pyCGM():
    def __init__(self, markers, measurements):
        self.marker_data = markers
        self.measurements = measurements
        self.frame = self.marker_data[0]

        self.measurement_keys = [measurement for measurement in self.measurements[0]]
        self.measurement_mapping = { measurement: index for index, measurement in enumerate(self.measurement_keys) }

        self.pelvis_joint_center = [np.NaN, np.NaN]

    def get_pelvis_joint_center(self):
        return self.pelvis_joint_center

    def marker(self, name):
        return self.frame[name] if name in self.frame else None

    def measurement(self, name):
        try:
            return self.measurements[1][self.measurement_mapping[name]]
        except KeyError:
            return None

    def find(self, name):
        try:
            ret = self.frame[name]
            return ret
        except KeyError:
            ret = self.measurement(name)
            return ret

    def run(self):
        for frame in self.marker_data:
            self.frame = frame
            self.calc_frames(frame)
    
    def calc_frames(self, frame):
        self.calc_angles(frame)

    def calc_angles(self, frame):
        self.calc_pelvis_joint_center(self, self.marker('RASI'), self.marker('LASI'), self.marker('RPSI'), self.marker('LPSI'), self.marker('SACR'))

    @staticmethod
    def calc_pelvis_joint_center(self, rasi, lasi, rpsi, lpsi, sacr):

        if rpsi is not None and lpsi is not None:
            sacrum = (rpsi + lpsi)/2.0
        else:
            sacrum = sacr

        origin = (rasi+lasi)/2.0
        beta1 = origin - sacrum
        beta2 = lasi - rasi
        y_axis = beta2 / np.linalg.norm(beta2)
        beta3_cal = np.dot(beta1, y_axis)
        beta3_cal2 = beta3_cal * y_axis
        beta3 = beta1-beta3_cal2
        x_axis = beta3/np.linalg.norm(beta3)
        z_axis = np.cross(x_axis, y_axis)
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin
        pelvis_axis = np.asarray([x_axis, y_axis, z_axis])
        pelvis = [origin, pelvis_axis, sacrum]

        self.pelvis_joint_center = pelvis
        return pelvis


class gonzCGM(pyCGM):

    @staticmethod
    @parameters('RASI', 'LASI', 'RPSI', 'LPSI', 'SACR', 'RKNE', 'LeftKneeWidth', 'RightKneeWidth')
    def calc_pelvis_joint_center(self, rasi, lasi, rpsi, lpsi, sacr, rkne, width_left_knee, width_right_knee):

        if rpsi is not None and lpsi is not None:
            sacrum = (rpsi + lpsi)/2.0
        else:
            sacrum = sacr

        origin = (rasi+lasi)/2.0
        beta1 = origin - sacrum
        beta2 = lasi - rasi
        y_axis = beta2 / np.linalg.norm(beta2)
        beta3_cal = np.dot(beta1, y_axis)
        beta3_cal2 = beta3_cal * y_axis
        beta3 = beta1-beta3_cal2
        x_axis = beta3/np.linalg.norm(beta3)
        z_axis = np.cross(x_axis, y_axis)
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin
        pelvis_axis = np.asarray([x_axis, y_axis, z_axis])
        pelvis = [origin, pelvis_axis, sacrum]

        self.pelvis_joint_center = pelvis
        return pelvis
