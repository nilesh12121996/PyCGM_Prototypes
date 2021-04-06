"""decorator approach"""

import functools

import numpy as np
import time


def markers(*args):
    def decorator(func):
        markers = list(args)

        @functools.wraps(func)
        def set_markers(self, *args):
            # print(self.mapping)
            data = [self.frame[self.mapping[key]] if key in self.mapping else None for key in markers]
            return func(*args, data)
        return set_markers
    return decorator


class CGM:
    def __init__(self, frame, mapping):
        self.frame = frame
        self._pelvis_axis = None
        self.mapping = {}
        for i, key in enumerate(mapping):
            self.mapping[key] = i

    @markers('RASI', 'LASI', 'RPSI', 'LPSI', 'SACR')
    def pelvisJointCenter(markers):
        # Get the Pelvis Joint Centre

        # REQUIRED MARKERS:
        # RASI
        # LASI
        # RPSI
        # LPSI

        rasi, lasi, rpsi, lpsi, sacr = markers

        if rpsi is not None and lpsi is not None:
            sacrum = (rpsi + lpsi)/2.0
        else:
            sacrum = sacr

        # REQUIRED LANDMARKS:
        # origin
        # sacrum

        # Origin is Midpoint between RASI and LASI
        origin = (rasi+lasi)/2.0

        beta1 = origin - sacrum
        beta2 = lasi - rasi

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
        time.sleep(0.0000001)
        self._pelvis_axis = self.pelvisJointCenter()

    @property
    def pelvis_axis(self):
        if self._pelvis_axis is None:
            self.run()
        return self._pelvis_axis


class ModCGM(CGM):
    @markers('RASI', 'LASI', 'RPSI', 'LPSI')
    def pelvisJointCenter(markers):
        # Get the Pelvis Joint Centre

        # REQUIRED MARKERS:
        # RASI
        # LASI
        # RPSI
        # LPSI

        rasi, lasi, rpsi, lpsi = markers

        if rpsi is not None and lpsi is not None:
            sacrum = (rpsi + lpsi)/2.0
        else:
            sacrum = 1

        # REQUIRED LANDMARKS:
        # origin
        # sacrum

        # Origin is Midpoint between RASI and LASI
        origin = (rasi+lasi)/2.0

        beta1 = origin - sacrum
        beta2 = lasi - rasi

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
