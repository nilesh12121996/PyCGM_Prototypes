from pyCGM import pyCGM
import numpy as np

class harrington_hip_CGM(pyCGM):

    # example hip_axis function using harrington pelvis joint center

    @pyCGM.parameters('InterAsisDistance', 'PD', 'Pelvis')
    def hip_axis(pelvis_width, pelvis_depth, pelvis_axis):
        # get the refactored 4x4 hip axis, using harrington hip joint center

        # unfortunately we don't have data with pelvis_depth
        return np.zeros((4,4))

        # what it would look like:
        l_hip_jc_x = -0.24 * pelvis_depth - 9.9
        l_hip_jc_y =  0.30 * pelvis_width + 10.9
        l_hip_jc_z = -0.33 * pelvis_width - 7.3
        l_hip_jc = [l_hip_jc_x, l_hip_jc_y, l_hip_jc_z]

        r_hip_jc_x = -0.24 * pelvis_depth - 9.9
        r_hip_jc_y = (0.30 * pelvis_width + 10.9) * -1
        r_hip_jc_z = -0.33 * pelvis_width - 7.3
        r_hip_jc = [r_hip_jc_x, r_hip_jc_y, r_hip_jc_z]

        hipaxis_center = (np.asarray(r_hip_jc) + np.asarray(l_hip_jc)) / 2

        pelvis_x_axis = pelvis_axis[0, :3]
        pelvis_y_axis = pelvis_axis[1, :3]
        pelvis_z_axis = pelvis_axis[2, :3]

        axis = np.zeros((4, 4))
        axis[3, 3] = 1.0
        axis[0, :3] = pelvis_x_axis
        axis[1, :3] = pelvis_y_axis
        axis[2, :3] = pelvis_z_axis
        axis[:3, 3] = hipaxis_center

        return axis

class oxfordCGM(pyCGM):

    # an example of a custom function that returns custom axis results

    @pyCGM.parameters('RANK', 'RCPG', 'RSTL', 'RHLX', 'RLCA', 'RHEE', 'RP1M', 'RP5M', 'RD5M', 'RTOE')
    def oxford_foot(rank, rcpg, rstl, rhlx, rlca, rhee, rp1m, rp5m, rd5m, rtoe):
        '''
        returns:
        LFF/RFF: L/R forefoot with respect to tibia
        LHF/RHF: L/R hindfoot with respect to tibia
        '''

        # show all parameters passed
        # print('Oxford called with:')
        # print('rank', rank, 'rcpg', rcpg,
        #       'rstl', rstl, 'rhlx', rhlx,
        #       'rlca', rlca, 'rhee', rhee,
        #       'rp1m', rp1m, 'rp5m', rp5m,
        #       'rd5m', rd5m, 'rtoe', rtoe)

        # calculations would take this shape
        lff = np.zeros((4,4))
        rff = np.zeros((4,4))
        lhf = np.ones((4,4))
        rhf = np.ones((4,4))

        return np.array([lff,rff,lhf,rhf])
