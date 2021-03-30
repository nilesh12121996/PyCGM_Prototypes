import functools

import numpy as np


def markers(*args):
    def decorator(func):
        markers = list(args)

        @functools.wraps(func)
        def set_markers(self, *args):
            data = [self.frame[key] if key in self.frame else None for key in markers]
            return func(*args, data)
        return set_markers
    return decorator


class CGM:
    def __init__(self, frame):
        self.frame = frame
        self._pelvis_axis = None

    @staticmethod
    def pelvisJointCenter(frame):
        # Get the Pelvis Joint Centre

        # REQUIRED MARKERS:
        # RASI
        # LASI
        # RPSI
        # LPSI

        RASI = frame['RASI']
        LASI = frame['LASI']

        if 'RPSI' in frame and 'LPSI' in frame:
            RPSI = frame['RPSI']
            LPSI = frame['LPSI']
            #  If no sacrum, mean of posterior markers is used as the sacrum
            sacrum = (RPSI+LPSI)/2.0
        else:
            sacrum = frame['SACR']

        # REQUIRED LANDMARKS:
        # origin
        # sacrum

        # Origin is Midpoint between RASI and LASI
        origin = (RASI+LASI)/2.0

        beta1 = origin - sacrum
        beta2 = LASI - RASI

        # Y_axis is normalized beta2
        y_axis = beta2 / np.linalg.norm(beta2)

        beta3_cal = np.dot(beta1, y_axis)
        beta3_cal2 = beta3_cal * y_axis
        beta3 = beta1-beta3_cal2
        x_axis = beta3/np.linalg.norm(beta3)

        # Z-axis is cross product of x_axis and y_axis.
        z_axis = np.cross(x_axis, y_axis)

        # Add the origin back to the vector
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin

        pelvis_axis = np.asarray([x_axis, y_axis, z_axis])

        pelvis = [origin, pelvis_axis, sacrum]

        return pelvis

    def run(self):
        # First Calculate Pelvis
        self._pelvis_axis = self.pelvisJointCenter(self.frame)
        return self.pelvis_axis

    @property
    def pelvis_axis(self):
        if self._pelvis_axis is None:
            self.run()
        return self._pelvis_axis
