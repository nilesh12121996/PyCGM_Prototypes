import numpy as np
from math import pi

class CalcAxes():

    def __init__(self):
        self.funcs = [self.pelvis_axis, self.hip_axis, self.knee_axis, self.ankle_axis, self.foot_axis,
                      self.head_axis, self.thorax_axis, self.clav_axis, self.hum_axis, self.rad_axis, self.hand_axis]

    def pelvis_axis(self, rasi, lasi, rpsi, lpsi, sacr=None):
        # get the refactored 4x4 pelvis axis

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

        # I added these back to check for correctness
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin

        pelvis = np.zeros((4, 4))
        pelvis[3, 3] = 1.0
        pelvis[0, :3] = x_axis
        pelvis[1, :3] = y_axis
        pelvis[2, :3] = z_axis
        pelvis[:3, 3] = origin

        return pelvis

    def hip_axis(self):
        return np.zeros((4,4))

    def knee_axis(self):
        return np.zeros((2,4,4))
        
    def ankle_axis(self):
        return np.zeros((2,4,4))
        
    def foot_axis(self):
        return np.zeros((2,4,4))
        
    def head_axis(self):
        return np.zeros((4,4))
        
    def thorax_axis(self):
        return np.zeros((4,4))
        
    def clav_axis(self):
        return np.zeros((2,4,4))
        
    def hum_axis(self):
        return np.zeros((2,4,4))
        
    def rad_axis(self):
        return np.zeros((2,4,4))

    def hand_axis(self):
        return np.zeros((2,4,4))

class CalcAngles():

    def __init__(self):
        self.funcs = [self.pelvis_angle, self.hip_angle, self.knee_angle, self.ankle_angle, self.foot_angle, self.head_angle,
                      self.thorax_angle, self.neck_angle, self.spine_angle, self.shoulder_angle, self.elbow_angle, self.wrist_angle]

    def pelvis_angle(self, axis_p, axis_d):
        
        beta = np.arctan2( (
            (axis_d[2][0] * axis_p[1][0])
            + (axis_d[2][1] * axis_p[1][1])
            + (axis_d[2][2] * axis_p[1][2])
        ),
        np.sqrt((
                axis_d[2][0] * axis_p[0][0]
                + axis_d[2][1] * axis_p[0][1]
                + axis_d[2][2] * axis_p[0][2]
            ) ** 2 + (
                axis_d[2][0] * axis_p[2][0]
                + axis_d[2][1] * axis_p[2][1]
                + axis_d[2][2] * axis_p[2][2]
            ) ** 2
        )
    )

        alpha = np.arctan2((
            (axis_d[2][0] * axis_p[0][0])
            + (axis_d[2][1] * axis_p[0][1])
            + (axis_d[2][2] * axis_p[0][2])
        ), (
            (axis_d[2][0] * axis_p[2][0])
            + (axis_d[2][1] * axis_p[2][1])
            + (axis_d[2][2] * axis_p[2][2])
        )
    )

        gamma = np.arctan2((
            (axis_d[0][0] * axis_p[1][0])
            + (axis_d[0][1] * axis_p[1][1])
            + (axis_d[0][2] * axis_p[1][2])
        ), (
            (axis_d[1][0] * axis_p[1][0])
            + (axis_d[1][1] * axis_p[1][1])
            + (axis_d[1][2] * axis_p[1][2])
        )
    )

        alpha = 180.0 * alpha / pi
        beta = 180.0 * beta / pi
        gamma = 180.0 * gamma / pi

        angle = [alpha, beta, gamma]
        return np.asarray(angle)

    def hip_angle(self):
        return np.zeros((2,3))

    def knee_angle(self):
        return np.zeros((2,3))

    def ankle_angle(self):
        return np.zeros((2,3))

    def foot_angle(self):
        return np.zeros((2,3))

    def head_angle(self):
        return np.zeros((3))

    def thorax_angle(self):
        return np.zeros((3))

    def neck_angle(self):
        return np.zeros((3))

    def spine_angle(self):
        return np.zeros((3))

    def shoulder_angle(self):
        return np.zeros((2,3))

    def elbow_angle(self):
        return np.zeros((2,3))

    def wrist_angle(self):
        return np.zeros((2,3))

