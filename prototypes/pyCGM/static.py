# -*- coding: utf-8 -*-
"""
This file provides helper functions for static calculations.

Created on Tue Jul 28 16:55:25 2015

@author: cadop
"""
import numpy as np


def get_static(motionData, vsk, flat_foot=False, GCS=None):
    """ Get Static Offset function

    Calculate the static offset angle values and return the values in radians

    Parameters
    ----------
    motionData : dict
                 Dictionary of marker lists.
    vsk : dict, optional
          Dictionary of various attributes of the skeleton.
    flat_foot : boolean, optional
                A boolean indicating if the feet are flat or not.
                The default value is False.
    GCS : array, optional
          An array containing the Global Coordinate System.
          If not provided, the default will be set to: [[1, 0, 0], [0, 1, 0],
          [0, 0, 1]].

    Returns
    -------
    calSM : dict
            Dictionary containing various marker lists of offsets.

    Examples
    --------
    >>> from pyCGM_Single.pycgmIO import loadC3D, loadVSK
    >>> from .static import get_static
    >>> import os
    >>> from pyCGM_Single.pyCGM_Helpers import getfilenames
    >>> fileNames=getfilenames(2)
    >>> c3dFile = fileNames[1]
    >>> vskFile = fileNames[2]
    >>> result = loadC3D(c3dFile)
    >>> data = result[0]
    >>> vskData = loadVSK(vskFile, False)
    >>> result = get_static(data,vskData,flat_foot=False)
    >>> result['Bodymass']
    75.0
    >>> result['RightKneeWidth']
    105.0
    >>> result['LeftTibialTorsion']
    0.0
    """
    static_offset = []
    head_offset = []
    IAD = []
    calSM = {}
    LeftLegLength = vsk['LeftLegLength']
    RightLegLength = vsk['RightLegLength']
    calSM['MeanLegLength'] = (LeftLegLength+RightLegLength)/2.0
    calSM['Bodymass'] = vsk['Bodymass']

    # Define the global coordinate system
    if not GCS:
        calSM['GCS'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    if vsk['LeftAsisTrocanterDistance'] and vsk['RightAsisTrocanterDistance']:
        calSM['L_AsisToTrocanterMeasure'] = vsk['LeftAsisTrocanterDistance']
        calSM['R_AsisToTrocanterMeasure'] = vsk['RightAsisTrocanterDistance']
    else:
        calSM['R_AsisToTrocanterMeasure'] = (0.1288 * RightLegLength) - 48.56
        calSM['L_AsisToTrocanterMeasure'] = (0.1288 * LeftLegLength) - 48.56

    if vsk['InterAsisDistance']:
        calSM['InterAsisDistance'] = vsk['InterAsisDistance']
    else:
        for frame in motionData:
            iadCalc = get_iad(frame['RASI'], frame['LASI'])
            IAD.append(iadCalc)
        InterAsisDistance = np.average(IAD)
        calSM['InterAsisDistance'] = InterAsisDistance

    try:
        calSM['RightKneeWidth'] = vsk['RightKneeWidth']
        calSM['LeftKneeWidth'] = vsk['LeftKneeWidth']

    except Exception:
        # no knee width
        calSM['RightKneeWidth'] = 0
        calSM['LeftKneeWidth'] = 0

    if not calSM['RightKneeWidth']:
        if 'RMKN' in list(motionData[0].keys()):
            # medial knee markers are available
            Rwidth = []
            Lwidth = []
            # average each frame
            for frame in motionData:
                RMKN = frame['RMKN']
                LMKN = frame['LMKN']

                RKNE = frame['RKNE']
                LKNE = frame['LKNE']

                Rdst = np.linalg.norm(RKNE - RMKN)
                Ldst = np.linalg.norm(LKNE - LMKN)
                Rwidth.append(Rdst)
                Lwidth.append(Ldst)

            calSM['RightKneeWidth'] = sum(Rwidth)/len(Rwidth)
            calSM['LeftKneeWidth'] = sum(Lwidth)/len(Lwidth)
    try:
        calSM['RightAnkleWidth'] = vsk['RightAnkleWidth']
        calSM['LeftAnkleWidth'] = vsk['LeftAnkleWidth']

    except Exception:
        # no knee width
        calSM['RightAnkleWidth'] = 0
        calSM['LeftAnkleWidth'] = 0

    if not calSM['RightAnkleWidth']:
        if 'RMKN' in motionData[0]:
            # medial knee markers are available
            Rwidth = []
            Lwidth = []
            # average each frame
            for frame in motionData:
                RMMA = frame['RMMA']
                LMMA = frame['LMMA']

                RANK = frame['RANK']
                LANK = frame['LANK']

                Rdst = np.linalg.norm(RMMA - RANK)
                Ldst = np.linalg.norm(LMMA - LANK)
                Rwidth.append(Rdst)
                Lwidth.append(Ldst)

            calSM['RightAnkleWidth'] = sum(Rwidth)/len(Rwidth)
            calSM['LeftAnkleWidth'] = sum(Lwidth)/len(Lwidth)

    calSM['RightTibialTorsion'] = vsk['RightTibialTorsion']
    calSM['LeftTibialTorsion'] = vsk['LeftTibialTorsion']

    calSM['RightShoulderOffset'] = vsk['RightShoulderOffset']
    calSM['LeftShoulderOffset'] = vsk['LeftShoulderOffset']

    calSM['RightElbowWidth'] = vsk['RightElbowWidth']
    calSM['LeftElbowWidth'] = vsk['LeftElbowWidth']
    calSM['RightWristWidth'] = vsk['RightWristWidth']
    calSM['LeftWristWidth'] = vsk['LeftWristWidth']

    calSM['RightHandThickness'] = vsk['RightHandThickness']
    calSM['LeftHandThickness'] = vsk['LeftHandThickness']

    for frame in motionData:
        pelvis_origin, pelvis_axis, sacrum = pelvis_joint_center(frame)
        hip = hip_joint_center(frame, pelvis_origin, pelvis_axis[0],
                               pelvis_axis[1], pelvis_axis[2], calSM)
        knee = knee_joint_center(frame, hip, 0, vsk=calSM)
        ankle = ankle_joint_center(frame, knee, 0, vsk=calSM)
        angle = get_static_angle(frame, ankle, knee, flat_foot, calSM)
        head = head_joint_center(frame)
        headangle = calculate_head_angle(head)

        static_offset.append(angle)
        head_offset.append(headangle)

    static = np.average(static_offset, axis=0)
    staticHead = np.average(head_offset)

    calSM['RightStaticRotOff'] = -static[0][0]
    calSM['RightStaticPlantFlex'] = static[0][1]
    calSM['LeftStaticRotOff'] = static[1][0]
    calSM['LeftStaticPlantFlex'] = static[1][1]
    calSM['HeadOffset'] = staticHead

    return calSM


def get_iad(rasi, lasi):
    r"""Get the Inter ASIS Distance (IAD)

    Calculates the Inter ASIS Distance from a given frame.

    Given the markers RASI and LASI, the Inter ASIS Distance is
    defined as:

    .. math::

        InterASISDist = \sqrt{(RASI_x-LASI_x)^2 + (RASI_y-LASI_y)^2 +
                        (RASI_z-LASI_z)^2}

    where :math:`RASI_x` is the x-coordinate of the RASI marker in frame.

    Parameters
    ----------
    rasi : array
        Array of marker data.
    lasi : array
        Array of marker data.

    Returns
    -------
    Inter Asis Distance : float
        The euclidian distance between the two markers.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import get_iad
    >>> rasi = np.array([395.37, 428.1, 1036.83])
    >>> lasi = np.array([183.19, 422.79, 1033.07])
    >>> np.around(get_iad(rasi, lasi), 2)
    212.28
    """

    return np.sqrt(np.sum([(rasi[i] - lasi[i])**2 for i in range(3)]))


def calculate_head_angle(head):
    r"""Static Head Calculation function

    This function first calculates the x, y, z axes of the head by
    subtracting the head origin from the given head axes. Then the offset
    angles are calculated. This is a y-rotation from the rotational matrix
    created by the matrix multiplication of the distal
    axis and the inverse of the proximal axis.

    The head axis is calculated as follows:

    ..math::

        head\_axis = \begin{bmatrix}
            head_{x1}-origin_x & head_{x2}-origin_y & head_{x3}-origin_z \\
            head_{y1}-origin_x & head_{y2}-origin_y & head_{y3}-origin_z \\
            head_{z1}-origin_x & head_{z2}-origin_y & head_{z3}-origin_z
            \end{bmatrix}\\

    The offset angle is defined as:

    .. math::

        \[ result = \arctan{\frac{M[0][2]}{M[2][2]}} \]

    where M is the rotation matrix produced from multiplying distal_axis and
    :math:`proximal_axis^{-1}`

    Parameters
    ----------
    head : array
        An array containing the head axis and head origin.

    Returns
    -------
    offset : float
        The head offset angle for static calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import calculate_head_angle
    >>> head = np.array([[[100.33, 83.39, 1484.08],
    ...        [98.97, 83.58, 1483.77],
    ...        [99.35, 82.64, 1484.76]],
    ...        [99.58, 82.79, 1483.8]], dtype=list)
    >>> np.around(calculate_head_angle(head), 2)
    0.28
    """

    # Calculate head_axis as above in the function description
    # [[head_axis_x1 - origin_x, ...], ...]
    head_axis = np.array([[head[0][y][x] - head[1][x] for x in range(3)]
                          for y in range(3)])

    # Inversion of [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # calculate_head_angle permanently assumes an incorrect axis.
    inverted_global_axis = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    # Calculate rotational matrix.
    rotation_matrix = np.matmul(head_axis, inverted_global_axis)

    # Return arctangent of angle y.
    with np.errstate(invalid='ignore', divide='ignore'):
        sine_y = rotation_matrix[0][2]
        cosine_y = np.nan_to_num(rotation_matrix[2][2])
        return np.arctan(sine_y/cosine_y)


def get_static_angle(frame, ankle_joint_center, knee_joint_center,
                     flat_foot, vsk=None):
    """Calculate the Static angle function

    Takes in anatomically uncorrected axis and anatomically correct axis.
    Corrects the axis depending on flat-footedness.

    Calculates the offset angle between those two axes.

    It is rotated from uncorrected axis in YXZ order.

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    ankle_joint_center : array
        An array containing the x,y,z axes marker positions of the ankle joint
        center.
    knee_joint_center : array
        An array containing the x,y,z axes marker positions of the knee joint
        center.
    flat_foot : boolean
        A boolean indicating if the feet are flat or not.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    angle : list
        Returns the offset angle represented by a 2x3x3 list.
        The array contains the right flexion, abduction, rotation angles
        (1x3x3) followed by the left flexion, abduction, rotation angles
        (1x3x3).

    Modifies
    --------
    The correct axis changes depending on the flat foot option.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import get_static_angle
    >>> frame = {'RTOE': np.array([427.95, 437.1,  41.77]),
    ...          'LTOE': np.array([175.79, 379.5,  42.61]),
    ...          'RHEE': np.array([406.46, 227.56,  48.76]),
    ...          'LHEE': np.array([223.6, 173.43,  47.93])}
    >>> ankle_joint_center = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...            np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.47, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> knee_joint_center = [np.array([364.18, 292.17, 515.19]),
    ...           np.array([143.55, 279.90, 524.78]),
    ...           np.array([[[364.65, 293.07, 515.19],
    ...           [363.29, 292.61, 515.04],
    ...           [364.05, 292.24, 516.18]],
    ...           [[143.66, 280.89, 524.63],
    ...           [142.56, 280.02, 524.86],
    ...           [143.65, 280.05, 525.77]]])]
    >>> flat_foot = True
    >>> vsk = { 'RightSoleDelta': 0.45,'LeftSoleDelta': 0.45 }
    >>> np.around(get_static_angle(frame, ankle_joint_center, knee_joint_center, flat_foot, vsk), 2)
    array([[-0.08,  0.2 , -0.65],
           [-0.67,  0.19, -0.32]])
    >>> flat_foot = False # Using the same variables and switching the flat_foot flag.
    >>> np.around(get_static_angle(frame, ankle_joint_center, knee_joint_center, flat_foot, vsk), 2)
    array([[ 0.03,  0.22, -0.16],
           [-0.48,  0.52,  0.28]])
    """

    # Get the each axis from the function.
    uncorrect = uncorrect_footaxis(frame, ankle_joint_center)

    # make the array which will be the input of findangle function
    right_uncorrect = np.vstack([np.subtract(uncorrect[2][0][x], uncorrect[0])
                                 for x in range(3)])
    left_uncorrect = np.vstack([np.subtract(uncorrect[2][1][x], uncorrect[1])
                                for x in range(3)])

    # Check if it is flat foot or not.
    if not flat_foot:
        rotaxis = rotaxis_non_flat_foot(frame, ankle_joint_center)[2]
    elif flat_foot:
        rotaxis = rotaxis_flat_foot(frame, ankle_joint_center, vsk=vsk)[2]

    # make the array to same format for calculating angle.
    right_axis = np.vstack([np.subtract(rotaxis[0][x], uncorrect[0])
                            for x in range(3)])
    left_axis = np.vstack([np.subtract(rotaxis[1][x], uncorrect[1])
                           for x in range(3)])

    angle = [get_ankle_angle(right_uncorrect, right_axis),
             get_ankle_angle(left_uncorrect, left_axis)]

    return angle


def pelvis_joint_center(frame):
    """Make the Pelvis Axis function

    Takes in a dictionary of x,y,z positions and marker names, as well as an
    index. Calculates the pelvis joint center and axis and returns both.

    Markers used: RASI,LASI,RPSI,LPSI
    Other landmarks used: origin, sacrum

    Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure
                   (ref. Kadaba 1990) and then normalized.
    Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
    Pelvis Z_axis: Cross product of x_axis and y_axis.

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.

    Returns
    -------
    pelvis : list
        Returns a list that contains the pelvis origin in a 1x3 array of xyz
        values, a 4x1x3 array composed of the pelvis x, y, z axes components,
        and the sacrum x, y, z position.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import pelvis_joint_center
    >>> frame = {'RASI': np.array([ 395.37,  428.1, 1036.83]),
    ...          'LASI': np.array([ 183.19,  422.79, 1033.07]),
    ...          'RPSI': np.array([ 341.42,  246.72, 1055.99]),
    ...          'LPSI': np.array([ 255.8,  241.42, 1057.3]) }
    >>> [np.around(arr, 2) for arr in pelvis_joint_center(frame)] #doctest: +NORMALIZE_WHITESPACE
    [array([ 289.28,  425.45, 1034.95]), array([[ 289.26,  426.44, 1034.83],
       [ 288.28,  425.42, 1034.93],
       [ 289.26,  425.56, 1035.94]]), array([ 298.61,  244.07, 1056.64])]
    """
    # Get the Pelvis Joint Centre
    RASI = frame['RASI']
    LASI = frame['LASI']

    try:
        RPSI = frame['RPSI']
        LPSI = frame['LPSI']
        #  If no sacrum, mean of posterior markers is used as the sacrum
        sacrum = (RPSI+LPSI)/2.0
    except Exception:
        pass  # going to use sacrum marker

    if 'SACR' in frame:
        sacrum = frame['SACR']

    # Origin is Midpoint between RASI and LASI
    origin = (RASI+LASI)/2.0

    # This calculate the each axis
    # beta1,2,3 is arbitrary name to help calculate.
    beta1 = origin-sacrum
    beta2 = LASI-RASI

    # Y_axis is normalized beta2
    beta2_norm = np.linalg.norm(beta2)
    if np.nan_to_num(beta2_norm):
        y_axis = np.divide(beta2, beta2_norm)
    else:
        y_axis = np.full_like(beta2, np.nan)

    # X_axis computed with a Gram-Schmidt orthogonalization procedure(ref.
    # Kadaba 1990) and then normalized.
    beta3_cal = np.dot(beta1, y_axis)
    beta3_cal2 = beta3_cal*y_axis
    beta3 = beta1-beta3_cal2
    beta3_norm = np.linalg.norm(beta3)
    if np.nan_to_num(beta3_norm):
        x_axis = np.divide(beta3, beta3_norm)
    else:
        x_axis = np.full_like(beta3, np.nan)

    # Z-axis is cross product of x_axis and y_axis.
    z_axis = np.cross(x_axis, y_axis)

    # Add the origin back to the vector
    y_axis = y_axis+origin
    z_axis = z_axis+origin
    x_axis = x_axis+origin

    pelvis_axis = np.asarray([x_axis, y_axis, z_axis])

    return [origin, pelvis_axis, sacrum]


def hip_joint_center(frame, pelvis_origin, pelvis_x, pelvis_y, pelvis_z,
                     vsk=None):
    """Calculate the hip joint center function.

    Takes in a dictionary of x,y,z positions and marker names, as well as an
    index. Calculates the hip joint center and returns the hip joint center.

    Other landmarks used: origin, sacrum
    Subject Measurement values used:
        MeanLegLength, R_AsisToTrocanterMeasure, InterAsisDistance,
        L_AsisToTrocanterMeasure

    Hip Joint Center: Computed using Hip Joint Center Calculation
    (ref. Davis_1991)

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    pelvis_origin : array
        An array of pelvis_origin, pelvis_x, pelvis_y, pelvis_z each x,y,z
        position.
    pelvis_x, pelvis_y, pelvis_z : int
        Respective axes of the pelvis.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    hip_JC : array
        Returns an array containing the left hip joint center x, y, z marker
        positions (1x3), followed by the right hip joint center x, y, z marker
        positions (1x3).

    Examples
    --------
    >>> import numpy as np
    >>> from .static import hip_joint_center
    >>> frame = None
    >>> vsk = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.51,
    ...        'L_AsisToTrocanterMeasure': 72.51, 'InterAsisDistance': 215.91}
    >>> pelvis_origin = [ 251.61, 391.74, 1032.89]
    >>> pelvis_x = [251.74, 392.73, 1032.79]
    >>> pelvis_y = [250.62, 391.87, 1032.87]
    >>> pelvis_z = [251.60, 391.85, 1033.89]
    >>> np.around(hip_joint_center(frame, pelvis_origin, pelvis_x, pelvis_y, pelvis_z, vsk), 2)    #doctest: +NORMALIZE_WHITESPACE
    array([[183.24, 338.8 , 934.65],
           [308.9 , 322.3 , 937.19]])
    """
    # Get Global Values=
    # Set the variables needed to calculate the joint angle
    mm = 7.0
    C = (vsk['MeanLegLength'] * 0.115) - 15.3
    theta = 0.500000178813934
    beta = 0.314000427722931
    half_inter_asis_distance = vsk['InterAsisDistance']/2.0

    asis_to_trocanter_measures = [vsk['R_AsisToTrocanterMeasure'],
                                  vsk['L_AsisToTrocanterMeasure']]
    # Hip Joint Center Calculation (ref. Davis_1991)
    pelvis_axis = [np.subtract(pelvis_x, pelvis_origin),
                   np.subtract(pelvis_y, pelvis_origin),
                   np.subtract(pelvis_z, pelvis_origin)]
    distances = []
    hip_joint_center = []
    for side in range(2):
        y_distance = C*np.sin(theta)-half_inter_asis_distance
        y_distance *= 1 if not side else -1
        # Calculate the distance to translate along the pelvis axis
        distances.append([(-asis_to_trocanter_measures[side] - mm)
                          * np.cos(beta) + C * np.cos(theta) * np.sin(beta),
                          y_distance,
                          (-asis_to_trocanter_measures[side] - mm)
                          * np.sin(beta) - C * np.cos(theta) * np.cos(beta)])

        # Multiply the distance to the unit pelvis axis
        pelvis_distances = [np.multiply(pelvis_axis[x], distances[side][x])
                            for x in range(3)]
        hip_joint_center.append(np.add([pelvis_distances[0][x]
                                        + pelvis_distances[1][x]
                                        + pelvis_distances[2][x]
                                for x in range(3)], pelvis_origin))

    return hip_joint_center[::-1]


def hip_axis_center(left_hip_joint_center, right_hip_joint_center,
                    pelvis_axis):
    """Calculate the hip joint axis function.

    Takes in a hip joint center of x,y,z positions as well as an index.
    and takes the hip joint center and pelvis origin/axis from previous
    functions. Calculates the hip axis and returns hip joint origin and axis.

    Hip center axis: mean at each x,y,z axis of the left and right hip joint
                     center.
    Hip axis: summation of the pelvis and hip center axes.

    Parameters
    ----------
    left_hip_joint_center, right_hip_joint_center: array
        Array of right_hip_joint_center and left_hip_joint_center each x,y,z
        position.
    pelvis_axis : array
        An array of pelvis origin and axis. The axis is also composed of 3
        arrays, each contain the x axis, y axis and z axis.

    Returns
    -------
    hip_axis_center, shared_hip_center: list
        Returns a list that contains the hip axis center in a 1x3 list of xyz
        values, which is then followed by a 3x2x3 list composed of the hip axis
        center x, y, and z axis components. The xyz axis components are 2x3
        lists consisting of the axis center in the first dimension and the
        direction of the axis in the second dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import hip_axis_center
    >>> right_hip_joint_center = [182.57, 339.43, 935.53]
    >>> left_hip_joint_center = [308.38, 322.80, 937.99]
    >>> pelvis_axis = [np.array([251.61, 391.74, 1032.89]),
    ...                np.array([[251.74, 392.73, 1032.79],
    ...                    [250.62, 391.87, 1032.87],
    ...                    [251.60, 391.85, 1033.89]]),
    ...                np.array([231.58, 210.25, 1052.25])]
    >>> [np.around(arr,8) for arr in hip_axis_center(left_hip_joint_center, right_hip_joint_center, pelvis_axis)] #doctest: +NORMALIZE_WHITESPACE
    [array([245.475, 331.115, 936.76 ]),
    array([[245.605, 332.105, 936.66 ],
           [244.485, 331.245, 936.74 ],
           [245.465, 331.225, 937.76 ]])]
    """

    # Get shared hip axis, it is inbetween the two hip joint centers
    hip_axis_center = np.mean([left_hip_joint_center, right_hip_joint_center],
                              axis=0)

    # Translate pelvis axis to shared hip centre
    # Add the origin back to the vector
    shared_hip_center = [np.add(np.subtract(pelvis_axis[1][x], pelvis_axis[0]),
                                hip_axis_center) for x in range(3)]

    return [hip_axis_center, shared_hip_center]


def knee_joint_center(frame, hip_joint_center, delta, vsk=None):
    """Calculate the knee joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, as well as an
    index and takes the hip axis and pelvis axis.
    Calculates the knee joint axis and returns the knee origin and axis.

    Markers used: RTHI, LTHI, RKNE, LKNE, hip_joint_center
    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait
    Analysis hand book, Baker2013)

    Parameters
    ----------
    frame : dict
        dictionary of marker lists.
    hip_joint_center : array
        An array of hip_joint_center containing the x,y,z axes marker positions
        of the hip joint center.
    delta : float
        The length from marker to joint center, retrieved from subject
        measurement file.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    R, L, knee_axes : list
        Returns a list that contains the knee axes' center in two 1x3 arrays of
        xyz values, which is then followed by a 2x3x3 array composed of the
        knee axis center x, y, and z axis components. The xyz axis components
        are 2x3 arrays consisting of the axis center in the first dimension and
        the direction of the axis in the second dimension.

    Modifies
    --------
    Delta is changed suitably to knee.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import knee_joint_center
    >>> vsk = { 'RightKneeWidth' : 105.0, 'LeftKneeWidth' : 105.0 }
    >>> frame = { 'RTHI': np.array([426.50, 262.65, 673.66]),
    ...           'LTHI': np.array([51.94, 320.02, 723.03]),
    ...           'RKNE': np.array([416.99, 266.23, 524.04]),
    ...           'LKNE': np.array([84.62, 286.69, 529.40])}
    >>> hip_joint_center = [[182.57, 339.43, 935.53],
    ...         [309.38, 322.80, 937.99]]
    >>> delta = 0
    >>> [np.around(arr, 2) for arr in knee_joint_center(frame,hip_joint_center,delta,vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([364.24, 292.34, 515.31]),
     array([143.55, 279.9 , 524.79]),
     array([[[364.69, 293.24, 515.31],
             [363.36, 292.78, 515.17],
             [364.12, 292.42, 516.3 ]],
            [[143.65, 280.88, 524.63],
             [142.56, 280.01, 524.86],
             [143.64, 280.04, 525.77]]])]
    """
    # Get Global Values
    hip_joint_center = [hip_joint_center[1], hip_joint_center[0]]
    thighs = [frame['RTHI'], frame['LTHI']]
    knees = [frame['RKNE'], frame['LKNE']]
    deltas = [(vsk['RightKneeWidth']/2.0)+7.0,
              (vsk['LeftKneeWidth']/2.0)+7.0]
    knee_axes = []
    axes = []
    positions = []
    for side in range(2):
        # Determine the position of knee_joint_center using the
        # find_joint_center function
        positions.append(find_joint_center(thighs[side],
                                           hip_joint_center[side],
                                           knees[side], deltas[side]))

        # Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        # Z axis is the thigh bone calculated by the hip_joint_center and the
        # knee_joint_center the axis is then normalized
        z_axis = np.subtract(hip_joint_center[side], positions[side])

        # X axis is perpendicular to the points plane which is determined by
        # KJC, HJC, KNE markers and calculated by each point's vector cross
        # vector. The axis is then normalized.
        if not side:
            x_axis = np.cross(z_axis, np.subtract(thighs[side], knees[side]))
        else:
            x_axis = np.cross(np.subtract(thighs[side], knees[side]), z_axis)

        # Y axis is determined by cross product of z_axis and x_axis.
        y_axis = np.cross(z_axis, x_axis)
        axes.append([x_axis, y_axis, z_axis])

        # Normalize the axes, right then left
        knee_axis = []
        for axis in axes[side]:
            # x_axis then y_axis and finally z_axis
            tmp_knee_axis = axis
            tmp_knee_axis_norm = np.nan_to_num(np.linalg.norm(tmp_knee_axis))
            if tmp_knee_axis_norm:
                tmp_knee_axis = np.divide(tmp_knee_axis, tmp_knee_axis_norm)
            knee_axis.append(tmp_knee_axis)

        # Add the origin back to the vector
        knee_axes.append(np.add(knee_axis, positions[side]))

    return [positions[0], positions[1], knee_axes]


def ankle_joint_center(frame, knee_joint_center, delta, vsk=None):
    """Calculate the ankle joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, an index
    and the knee axis.
    Calculates the ankle joint axis and returns the ankle origin and axis.

    Markers used: RTIB, LTIB, RANK, LANK, knee_joint_center
    Subject Measurement values used: RightKneeWidth, LeftKneeWidth

    Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait
    Analysis hand book, Baker2013).

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    knee_joint_center : array
        An array of knee_joint_center each x,y,z position.
    delta : float
        The length from marker to joint center, retrieved from subject
        measurement file.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    R, L, axis : list
        Returns a list that contains the ankle axis origin in 1x3 arrays of xyz
        values and a 3x2x3 list composed of the ankle origin, x, y, and z axis
        components. The xyz axis components are 2x3 lists consisting of the
        origin in the first dimension and the direction of the axis in the
        second dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import ankle_joint_center
    >>> vsk = { 'RightAnkleWidth' : 70.0, 'LeftAnkleWidth' : 70.0,
    ...         'RightTibialTorsion': 0.0, 'LeftTibialTorsion' : 0.0}
    >>> frame = { 'RTIB': np.array([433.98, 211.93, 273.30]),
    ...           'LTIB': np.array([50.04, 235.91, 364.32]),
    ...           'RANK': np.array([422.77, 217.74, 92.86]),
    ...           'LANK': np.array([58.57, 208.55, 86.17]) }
    >>> knee_joint_center = [np.array([364.18, 292.17, 515.19]),
    ...           np.array([143.55, 279.90, 524.78]),
    ...           np.array([[[364.65, 293.07, 515.18],
    ...           [363.29, 292.61, 515.04],
    ...           [364.05, 292.24, 516.18]],
    ...           [[143.66, 280.89, 524.63],
    ...           [142.56, 280.02, 524.86],
    ...            [143.65, 280.05, 525.77]]])]
    >>> delta = 0
    >>> [np.around(arr, 2) for arr in ankle_joint_center(frame, knee_joint_center, delta, vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([393.76, 247.68,  87.74]),
     array([ 98.75, 219.47,  80.63]),
     array([[[394.48, 248.37,  87.71],
             [393.07, 248.39,  87.61],
             [393.69, 247.78,  88.73]],
            [[ 98.47, 220.43,  80.53],
             [ 97.79, 219.21,  80.76],
             [ 98.85, 219.6 ,  81.62]]])]
    """

    # Get Global Values
    tibias = [frame['RTIB'], frame['LTIB']]
    ankles = [frame['RANK'], frame['LANK']]
    deltas = [(vsk['RightAnkleWidth']/2.0)+7.0,
              (vsk['LeftAnkleWidth']/2.0)+7.0]
    torsions = [np.radians(vsk['RightTibialTorsion']),
                np.radians(vsk['LeftTibialTorsion'])]
    rotated_axes = []
    axes = []
    positions = []
    # This is Torsioned Tibia and this describe the ankle angles
    # Tibial frontal plane being defined by ANK,TIB and KJC
    for side in range(2):
        # Determine the position of ankle_joint_center using the
        # find_joint_center function
        positions.append(find_joint_center(tibias[side],
                                           knee_joint_center[side],
                                           ankles[side], deltas[side]))
        # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book,
        # Baker2013). The Z axis is shank bone calculated by the
        # ankle_joint_center and the knee_joint_center.
        z_axis = np.subtract(knee_joint_center[side], positions[side])

        # X axis is perpendicular to the points plane which is determined by
        # the ANK, TIB and KJC markers and calculated by each point's vector
        # cross vector. The x_axis vector is making a tibia plane to be assumed
        # as a rigid segment.
        if not side:
            x_axis = np.cross(z_axis, np.subtract(tibias[side], ankles[side]))
        else:
            x_axis = np.cross(np.subtract(tibias[side], ankles[side]), z_axis)

        # Y axis is determined by cross product of axis_z and axis_x.
        y_axis = np.cross(z_axis, x_axis)
        axes.append([x_axis, y_axis, z_axis])

        # Normalize the axes, right then left
        ankle_axis = []
        for axis in axes[side]:
            # x_axis then y_axis and finally z_axis
            tmp_ankle_axis = axis
            tmp_ankle_axis_norm = np.nan_to_num(np.linalg.norm(tmp_ankle_axis))
            if tmp_ankle_axis_norm:
                tmp_ankle_axis = np.divide(tmp_ankle_axis, tmp_ankle_axis_norm)
            ankle_axis.append(tmp_ankle_axis)

        # Rotate the axes about the tibia torsion.
        cosine = np.cos(torsions[side])
        sine = np.sin(torsions[side])
        rotated_axis = [np.subtract(cosine * ankle_axis[0],
                                    sine * ankle_axis[1]),
                        np.add(sine * ankle_axis[0], cosine * ankle_axis[1]),
                        ankle_axis[2]]
        # Add the origin back to the vector
        rotated_axes.append(np.add(rotated_axis, positions[side]))

    # Both of axis in array.
    return [positions[0], positions[1], rotated_axes]


def foot_joint_center(rtoe, ltoe, static_info, ankle_joint_center):
    r"""Calculate the foot joint center and axis function.

    Takes in a dictionary of xyz positions and marker names, the ankle axis and
    knee axis.
    Calculates the foot joint axis by rotating the incorrect foot joint axes
    about the offset angle.
    Returns the foot axis origin and axis.

    In the case of the foot joint center, we've already made 2 kinds of axes
    for the static offset angle and then, we call this static offset angle as
    an input of this function for the dynamic trial.

    Special Cases:

    (anatomically uncorrected foot axis)
    If flat foot, make the reference markers instead of HEE marker whose height
    is the same as TOE marker's height. Else use the HEE marker for making Z
    axis.

    Markers used: RTOE,LTOE
    Other landmarks used: ANKLE_FLEXION_AXIS
    Subject Measurement values used: RightStaticRotOff, RightStaticPlantFlex,
    LeftStaticRotOff, LeftStaticPlantFlex

    The incorrect foot joint axes for both feet are calculated using the
    following calculations:
        z-axis = ankle joint center - TOE marker
        y-flex = ankle joint center flexion - ankle joint center
        x-axis = y-flex cross z-axis
        y-axis = z-axis cross x-axis
    Calculate the foot joint axis by rotating incorrect foot joint axes
    about offset angle.

    .. math::

        z_{axis} = ankle\_joint\_center - toe\_marker\\
        flexion\_axis = ankle\_joint\_center\_flexion - ankle\_joint\_center\\
        x_{axis} = flexion\_axis \times z_{axis}\\
        y_{axis} = z_{axis} \times x_{axis}\\
        \\
        \text{Rotated about the } y_{axis} \text{:}\\
        rotation = \begin{bmatrix}
        cos(\beta) * x_{axis}[0] + sin(\beta) * z_{axis}[0] & cos(\beta)
        * x_{axis}[1] + sin(\beta) * z_{axis}[1] & cos(\beta) * x_{axis}[2]
        + sin(\beta) * z_{axis}[2\\
        y_{axis}[0] & y_{axis}[1] & y_{axis}[2]\\
        -sin(\beta) * x_{axis}[0] + cos(\beta) * z_{axis}[0] & -sin(\beta)
        * x_{axis}[1] + cos(\beta) * z_{axis}[1] & -sin(\beta) * x_{axis}[2]
        + cos(\beta) * z_{axis}[2]
        \end{bmatrix}\\
        \\
        \text{Rotated about the } x_{axis} \text{:}\\
        rotation = \begin{bmatrix}
        cos(\alpha) * rotation[1][0] - sin(\alpha) * rotation[2][0]
        & cos(\alpha) * rotation[1][1] - sin(\alpha) * rotation[2][1]
        & cos(\alpha) * rotation[1][2] - sin(\alpha) * rotation[2][2]\\
        y_{axis}[0] & y_{axis}[1] & y_{axis}[2]\\
        sin(\alpha) * rotation[1][0] + cos(\alpha) * rotation[2][0]
        & sin(\alpha) * rotation[1][1] + cos(\alpha) * rotation[2][1]
        & sin(\alpha) * rotation[1][2] + cos(\alpha) * rotation[2][2]
        \end{bmatrix}

    Parameters
    ----------
    rtoe : array
        Array of marker data.
    ltoe : array
        Array of marker data.
    static_info : array
        An array containing offset angles.
    ankle_joint_center : array
        An array containing the x,y,z axes marker positions of the
        ankle joint center.

    Returns
    -------
    rtoe, ltoe, foot_axis : array
        Returns a list that contain the toe (right and left) markers in
        1x3 arrays of xyz values and a 2x3x3 array composed of the foot axis
        center x, y, and z axis components. The xyz axis components are 2x3
        arrays consisting of the axis center in the first dimension and the
        direction of the axis in the second dimension.

    Modifies
    --------
    Axis changes the following in the static info.

    You can set the static_info with the button and this will calculate the
    offset angles.
    The first setting, the foot axis shows the uncorrected foot anatomical
    reference axis(Z_axis point to the AJC from TOE).

    If you press the static_info button so if static_info is not None,
    then the static offset angles are applied to the reference axis.
    The reference axis is Z axis point to HEE from TOE

    Examples
    --------
    >>> import numpy as np
    >>> from .static import foot_joint_center
    >>> rtoe = np.array([442.82, 381.62, 42.66])
    >>> ltoe = np.array([39.44, 382.45, 41.79])
    >>> static_info = [[0.03, 0.15, 0],
    ...                [0.01, 0.02, 0]]
    >>> ankle_joint_center = [np.array([393.76, 247.68, 87.74]),
    ...                       np.array([98.75, 219.47, 80.63]),
    ...                       [[np.array([394.48, 248.37, 87.72]),
    ...                         np.array([393.07, 248.39, 87.62]),
    ...                         np.array([393.69, 247.78, 88.73])],
    ...                        [np.array([98.47, 220.43, 80.53]),
    ...                         np.array([97.79, 219.21, 80.76]),
    ...                         np.array([98.85, 219.60, 81.62])]]]
    >>> delta = 0
    >>> [np.around(arr,2) for arr in foot_joint_center(rtoe, ltoe, static_info, ankle_joint_center)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]),
     array([ 39.44, 382.45,  41.79]),
     array([[[442.89, 381.76,  43.65],
            [441.89, 382.  ,  42.67],
            [442.45, 380.7 ,  42.82]],
           [[ 39.51, 382.68,  42.76],
            [ 38.5 , 382.15,  41.93],
            [ 39.76, 381.53,  41.99]]])]
    """
    toes = [rtoe, ltoe]

    # Dealing with the Incorrect Axis
    foot_axis = []
    for side in range(2):  # calculate the left and right foot axes
        # Z_axis taken from toe marker to Ankle Joint Center; then normalized.
        z_axis = np.array(ankle_joint_center[side]) - np.array(toes[side])
        z_axis_norm = np.nan_to_num(np.linalg.norm(z_axis))
        if z_axis_norm:
            z_axis = np.divide(z_axis, z_axis_norm)

        # Calculated flexion axis from Ankle Joint Center; then normalized.
        flexion_axis = (np.array(ankle_joint_center[2][side][1])
                        - np.array(ankle_joint_center[side]))
        flexion_axis_norm = np.nan_to_num(np.linalg.norm(flexion_axis))
        if flexion_axis_norm:
            flexion_axis = np.divide(flexion_axis, flexion_axis_norm)

        # X_axis taken from cross product of Z_axis and the flexion axis.
        x_axis = np.cross(flexion_axis, z_axis)
        x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
        if x_axis_norm:
            x_axis = np.divide(x_axis, x_axis_norm)

        # Y_axis take from cross product of Z_axis and X_axis (perpendicular).
        # Then normalized.
        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
        if y_axis_norm:
            y_axis = np.divide(y_axis, y_axis_norm)

        # Apply static offset angle to the incorrect foot axes
        # static offset angle are taken from static_info variable in radians.
        # N.B. This replaces a procedure that converted from radians to degrees
        # and back to radians, the result was then rounded to 5 decimal places.
        alpha = static_info[side][0]
        beta = static_info[side][1]

        # Rotate incorrect foot axis around y axis first.
        rotated_axis = [[(np.cos(beta) * x_axis[x] + np.sin(beta) * z_axis[x])
                         for x in range(3)], y_axis,
                        [(-np.sin(beta) * x_axis[x] + np.cos(beta) * z_axis[x])
                         for x in range(3)]]

        # Rotate incorrect foot axis around x axis next.
        rotated_axis = [rotated_axis[0],
                        [(np.cos(alpha) * rotated_axis[1][x] - np.sin(alpha)
                         * rotated_axis[2][x]) for x in range(3)],
                        [(np.sin(alpha) * rotated_axis[1][x] + np.cos(alpha)
                         * rotated_axis[2][x]) for x in range(3)]]

        # Attach each axis to the origin
        foot_axis.append([np.array(axis) + np.array(toes[side])
                          for axis in rotated_axis])

    return [rtoe, ltoe, foot_axis]


def head_joint_center(frame):
    """Calculate the head joint axis function.

    Takes in a dictionary of x,y,z positions and marker names.
    Calculates the head joint center and returns the head joint center and
    axis.

    Markers used: LFHD, RFHD, LBHD, RBHD

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.

    Returns
    -------
    head_axis, origin : list
        Returns a list containing a 1x3x3 list containing the x, y, z axis
        components of the head joint center and a 1x3 list containing the
        head origin x, y, z position.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import head_joint_center
    >>> frame = {'RFHD': np.array([325.83, 402.55, 1722.5]),
    ...          'LFHD': np.array([184.55, 409.69, 1721.34]),
    ...          'RBHD': np.array([304.4, 242.91, 1694.97]),
    ...          'LBHD': np.array([197.86, 251.29, 1696.90])}
    >>> [np.around(arr, 2) for arr in head_joint_center(frame)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 255.35,  405.15, 1721.76],
            [ 256.1 ,  405.71, 1721.86],
            [ 255.93,  406.79, 1722.03]]), array([ 255.19,  406.12, 1721.92])]
    """

    # Get the marker positions used for joint calculation
    LFHD = frame['LFHD']
    RFHD = frame['RFHD']
    LBHD = frame['LBHD']
    RBHD = frame['RBHD']

    # get the midpoints of the head to define the sides
    front = [np.mean([LFHD[index], RFHD[index]]) for index in range(3)]
    back = [np.mean([LBHD[index], RBHD[index]]) for index in range(3)]
    left = [np.mean([LBHD[index], LFHD[index]]) for index in range(3)]
    right = [np.mean([RFHD[index], RBHD[index]]) for index in range(3)]

    # Get the vectors from the sides with primary x axis facing front
    # First get the x direction
    x_axis = np.subtract(front, back)
    x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
    if x_axis_norm:
        x_axis = np.divide(x_axis, x_axis_norm)

    # get the direction of the y axis
    y_axis = np.subtract(left, right)
    y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
    if y_axis_norm:
        y_axis = np.divide(y_axis, y_axis_norm)

    # get z axis by cross-product of x axis and y axis.
    z_axis = np.subtract(x_axis, y_axis)
    z_axis_norm = np.nan_to_num(np.linalg.norm(z_axis))
    if z_axis_norm:
        z_axis = np.divide(z_axis, z_axis_norm)

    # make sure all x,y,z axis is orthogonal each other by cross-product
    y_axis = np.subtract(z_axis, x_axis)
    y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
    if y_axis_norm:
        y_axis = np.divide(y_axis, y_axis_norm)
    x_axis = np.subtract(y_axis, z_axis)
    x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
    if x_axis_norm:
        x_axis = np.divide(x_axis, x_axis_norm)

    # Add the origin back to the vector to get it in the right position
    head_axis = [np.add(axis, front) for axis in [x_axis, y_axis, z_axis]]

    result = [head_axis, front]
    return result


def uncorrect_footaxis(frame, ankle_joint_center):
    """Calculate the anatomically uncorrected foot joint center and axis
    function.

    Takes in a dictionary of xyz positions and marker names and takes the ankle
    axis.
    Calculate the anatomical uncorrect foot axis.

    Markers used: RTOE, LTOE

    Given a marker RTOE and the ankle JC, the right anatomically incorrect foot
    axis is calculated with:

    .. math::
        R = [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

    where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten
    from frame['RTOE']

    :math:`R_x` is the unit vector of :math:`Yflex_R \times R_z`

    :math:`R_y` is the unit vector of :math:`R_z \times R_x`

    :math:`R_z` is the unit vector of the axis from right toe to right ankle
    joint center

    :math:`Yflex_R` is the unit vector of the axis from right ankle flexion
    to right ankle joint center

    The same calculation applies for the left anatomically incorrect foot axis
    by replacing all the right values with left values

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    ankle_joint_center : array
        An array of ankle_joint_center each x,y,z position.

    Returns
    -------
    R, L, foot_axis : list
        Returns a list representing the incorrect foot joint center, the list
        contains two 1x3 arrays representing the foot axis origin x, y, z
        positions and a 3x2x3 list containing the foot axis center in the first
        dimension and the direction of the axis in the second dimension. This
        will be used for calculating static offset angle in static calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import uncorrect_footaxis
    >>> frame = { 'RTOE': [442.82, 381.62, 42.66],
    ...           'LTOE': [39.44, 382.45, 41.79]}
    >>> ankle_joint_center = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...            np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.47, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> [np.around(arr, 2) for arr in uncorrect_footaxis(frame,ankle_joint_center)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]),
    array([ 39.44, 382.45,  41.79]),
    array([[[442.94, 381.9 ,  43.61],
            [441.88, 381.97,  42.68],
            [442.49, 380.72,  42.96]],
           [[ 39.5 , 382.7 ,  42.76],
            [ 38.5 , 382.14,  41.93],
            [ 39.77, 381.53,  42.01]]])]
    """
    # Get global values
    toes = [frame['RTOE'], frame['LTOE']]
    foot_axes = []
    for side in range(2):
        # z axis is from Toe to AJC and normalized.
        z_axis = np.subtract(ankle_joint_center[side], toes[side])
        z_axis_norm = np.nan_to_num(np.linalg.norm(z_axis))
        if z_axis_norm:
            z_axis = np.divide(z_axis, z_axis_norm)

        # Bring flexion axis from ankle axis.
        flexion_axis = np.subtract(ankle_joint_center[2][side][1],
                                   ankle_joint_center[side])
        flexion_axis_norm = np.nan_to_num(np.linalg.norm(flexion_axis))
        if flexion_axis_norm:
            flexion_axis = np.divide(flexion_axis, flexion_axis_norm)

        # Calculate each x,y,z axis of foot using cross-product and make sure
        # x,y,z axis is orthogonal each other.
        x_axis = np.cross(flexion_axis, z_axis)
        x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
        if x_axis_norm:
            x_axis = np.divide(x_axis, x_axis_norm)

        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
        if y_axis_norm:
            y_axis = np.divide(y_axis, y_axis_norm)

        foot_axes.append([np.add(axis, toes[side])
                          for axis in [x_axis, y_axis, z_axis]])

    return [toes[0], toes[1], foot_axes]


def rotaxis_flat_foot(frame, ankle_joint_center, vsk=None):
    r"""Calculate the anatomically correct foot joint center and axis function
    for a flat foot.

    Takes in a dictionary of xyz positions and marker names and the ankle axis
    then calculates the anatomically correct foot axis for a flat foot.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Given the right ankle_joint_center and the markers :math:`TOE_R` and
    :math:`HEE_R`, the right anatomically correct foot axis is calculated with:

    .. math::
        R = [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

    where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten
    from frame['RTOE']

    :math:`R_x` is the unit vector of
    :math:`(AnkleFlexion_R - AnkleJC_R) \times R_z`

    :math:`R_y` is the unit vector of :math:`R_z \times R_x`

    :math:`R_z` is the unit vector of
    :math:`(A \times (HEE_R - TOE_R)) \times A`

    A is the unit vector of
    :math:`(HEE_R - TOE_R) \times (AnkleJC_R - TOE_R)`

    The same calculation applies for the left anatomically correct foot axis by
    replacing all the right values with left values.

    Parameters
    ----------
    frame : array
        Dictionary of marker lists.
    ankle_joint_center : array
        An array of ankle_joint_center each x,y,z position.
    vsk : dict, optional
        A dictionary containing subject measurements from a VSK file.

    Returns
    -------
    rtoe, ltoe, foot_axes: list
        Returns a list representing the correct foot joint center for a flat
        foot, the list contains 2 1x3 arrays representing the foot axis origin
        x, y, z positions and a 3x2x3 list containing the foot axis center in
        the first dimension and the direction of the axis in the second
        dimension.

    Modifies
    --------
    If the subject wears shoe, Soledelta is applied. then axes are changed
    following Soledelta.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import rotaxis_flat_foot
    >>> frame = {'RHEE': [374.01, 181.58, 49.51],
    ...          'LHEE': [105.30, 180.21, 47.16],
    ...          'RTOE': [442.82, 381.62, 42.66],
    ...          'LTOE': [39.44, 382.45, 41.79]}
    >>> ankle_joint_center = [np.array([393.76, 247.68, 87.74]),
    ...                       np.array([98.75, 219.47, 80.63]),
    ...                       [[np.array([394.48, 248.37, 87.72]),
    ...                         np.array([393.07, 248.39, 87.62]),
    ...                         np.array([393.69, 247.78, 88.73])],
    ...                         [np.array([98.48, 220.43, 80.53]),
    ...                          np.array([97.79, 219.21, 80.76]),
    ...                          np.array([98.85, 219.60, 81.62])]]]
    >>> vsk = { 'RightSoleDelta': 0.45, 'LeftSoleDelta': 0.45}
    >>> [np.around(arr, 2) for arr in rotaxis_flat_foot(frame, ankle_joint_center, vsk)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]),
     array([ 39.44, 382.45,  41.79]),
     array([[[442.33, 381.82,  43.51],
             [442.01, 381.88,  42.13],
             [442.49, 380.67,  42.69]],
            [[ 39.14, 382.38,  42.74],
             [ 38.54, 382.15,  41.48],
             [ 39.75, 381.5 ,  41.82]]])]
    """
    # Get Global Values
    toes = [frame['RTOE'], frame['LTOE']]
    heels = [frame['RHEE'], frame['LHEE']]
    sole_deltas = [vsk['RightSoleDelta'], vsk['LeftSoleDelta']]
    foot_axes = []
    for side in range(2):
        # Toe axis's origin is marker position of TOE
        ankle_joint_center[side][2] += sole_deltas[side]

        # For foot flat, Z axis pointing same height of TOE marker from TOE to
        # the Ankle Joint Center
        heel_to_toe = np.subtract(heels[side], toes[side])
        heel_to_toe_norm = np.nan_to_num(np.linalg.norm(heel_to_toe))
        if heel_to_toe_norm:
            heel_to_toe = np.divide(heel_to_toe, heel_to_toe_norm)

        # Bring flexion axis from ankle axis.
        flexion_axis = np.subtract(ankle_joint_center[2][side][1],
                                   ankle_joint_center[side])
        flexion_axis_norm = np.nan_to_num(np.linalg.norm(flexion_axis))
        if flexion_axis_norm:
            flexion_axis = np.divide(flexion_axis, flexion_axis_norm)

        # Calculate each x,y,z axis of foot using cross-product and make sure
        # x,y,z axis is orthogonal each other.
        x_axis = np.cross(flexion_axis, heel_to_toe)
        x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
        if x_axis_norm:
            x_axis = np.divide(x_axis, x_axis_norm)

        y_axis = np.cross(heel_to_toe, x_axis)
        y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
        if y_axis_norm:
            y_axis = np.divide(y_axis, y_axis_norm)

        z_axis = np.cross(x_axis, y_axis)
        z_axis_norm = np.nan_to_num(np.linalg.norm(z_axis))
        if z_axis_norm:
            z_axis = np.divide(z_axis, z_axis_norm)

        foot_axes.append([np.add(axis, toes[side])
                          for axis in [x_axis, y_axis, z_axis]])

    return [toes[0], toes[1], foot_axes]


def rotaxis_non_flat_foot(frame, ankle_joint_center):
    """Calculate the anatomically correct foot joint center and axis function
    for a non-flat foot.

    Takes in a dictionary of xyz positions & marker names and the ankle axis
    then calculates the anatomically correct foot axis for a non-flat foot.

    Markers used: RTOE, LTOE, RHEE, LHEE

    Given the right ankle joint center and the markers :math:`TOE_R` and
    :math:`HEE_R , the right anatomically correct foot axis is calculated
    with:

    .. math::
    R is [R_x + ROrigin_x, R_y + ROrigin_y, R_z + ROrigin_z]

    where :math:`ROrigin_x` is the x coor of the foot axis's origin gotten
    from frame['RTOE']

    :math:`R_x` is the unit vector of :math:`YFlex_R \times R_z`

    :math:`R_y` is the unit vector of :math:`R_z \times R_x`

    :math:`R_z` is the unit vector of :math:`(HEE_R - TOE_R)`

    :math:`YFlex_R` is the unit vector of
    :math:`(AnkleFlexion_R - AnkleJC_R)`

    The same calculation applies for the left anatomically correct foot
    axis by replacing all the right values with left values.

    Parameters
    ----------
    frame : dict
        Dictionary of marker lists.
    ankle_joint_center : array
        An array of ankle_joint_center each x,y,z position.

    Returns
    -------
    R, L, foot_axis: list
        Returns a list representing the correct foot joint center for a
        non-flat foot, the list contains two 1x3 arrays representing the foot
        axis origin x, y, z positions and a 3x2x3 list containing the foot axis
        center in the first dimension and the direction of the axis in the
        second dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import rotaxis_non_flat_foot
    >>> frame = { 'RHEE': [374.01, 181.58, 49.51],
    ...            'LHEE': [105.30, 180.21, 47.16],
    ...            'RTOE': [442.82, 381.62, 42.66],
    ...            'LTOE': [39.44, 382.45, 41.79]}
    >>> ankle_joint_center = [np.array([393.76, 247.68, 87.74]),
    ...            np.array([98.75, 219.47, 80.63]),
    ...            [[np.array([394.48, 248.37, 87.72]),
    ...            np.array([393.07, 248.39, 87.62]),
    ...            np.array([393.69, 247.78, 88.73])],
    ...            [np.array([98.47, 220.43, 80.53]),
    ...            np.array([97.79, 219.21, 80.76]),
    ...            np.array([98.85, 219.60, 81.62])]]]
    >>> [np.around(arr, 2) for arr in rotaxis_non_flat_foot(frame, ankle_joint_center)] #doctest: +NORMALIZE_WHITESPACE
    [array([442.82, 381.62,  42.66]),
     array([ 39.44, 382.45,  41.79]),
     array([[[442.72, 381.69,  43.65],
             [441.88, 381.94,  42.54],
             [442.49, 380.67,  42.69]],
            [[ 39.56, 382.51,  42.78],
             [ 38.5 , 382.15,  41.92],
             [ 39.75, 381.5 ,  41.82]]])]
    """
    # Get global values
    toes = [frame['RTOE'], frame['LTOE']]
    heels = [frame['RHEE'], frame['LHEE']]
    foot_axes = []
    for side in range(2):
        # For foot flat, Z axis pointing same height of TOE marker from TOE to
        # to the Ankle Joint Center
        heel_to_toe = np.subtract(heels[side], toes[side])
        heel_to_toe_norm = np.nan_to_num(np.linalg.norm(heel_to_toe))
        if heel_to_toe_norm:
            heel_to_toe = np.divide(heel_to_toe, heel_to_toe_norm)

        # Bring flexion axis from ankle axis.
        flexion_axis = np.subtract(ankle_joint_center[2][side][1],
                                   ankle_joint_center[side])
        flexion_axis_norm = np.nan_to_num(np.linalg.norm(flexion_axis))
        if flexion_axis_norm:
            flexion_axis = np.divide(flexion_axis, flexion_axis_norm)

        # Calculate each x,y,z axis of foot using cross-product and make sure
        # x,y,z axis is orthogonal each other.
        x_axis = np.cross(flexion_axis, heel_to_toe)
        x_axis_norm = np.nan_to_num(np.linalg.norm(x_axis))
        if x_axis_norm:
            x_axis = np.divide(x_axis, x_axis_norm)

        y_axis = np.cross(heel_to_toe, x_axis)
        y_axis_norm = np.nan_to_num(np.linalg.norm(y_axis))
        if y_axis_norm:
            y_axis = np.divide(y_axis, y_axis_norm)

        z_axis = heel_to_toe

        foot_axes.append([np.add(axis, toes[side])
                          for axis in [x_axis, y_axis, z_axis]])

    return [toes[0], toes[1], foot_axes]


def get_ankle_angle(proximal_axis, distal_axis):
    r"""Static angle calculation function.

    This function takes in two axes and returns three angles.
    It uses an inverse Euler rotation matrix in YXZ order.
    The output shows the angle in degrees.

    Since we use arc tangent we must check if the angle is in area between
    -pi/2 and pi/2 but because the static offset angle under pi/2, it doesn't
    matter.

    The alpha, beta, and gamma angles are defined as:
    .. math::
        \[ alpha = \arctan{\frac{M[2][1]}{\sqrt{M[2][0]^2 + M[2][2]^2}}} \]
        \[ beta = \arctan{\frac{-M[2][0]}{M[2][2]}} \]
        \[ gamma = \arctan{\frac{-M[0][1]}{M[1][1]}} \]
    where M is the rotation matrix produced from multiplying distal_axis
    and :math:`proximal_axis^{-1}`

    Parameters
    ----------
    proximal_axis : list
        Shows the unit vector of proximal_axis, the position of the proximal
        axis.
    distal_axis : list
        Shows the unit vector of distal_axis, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the alpha, beta, gamma angles in degrees in a 1x3 corresponding
        list.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import get_ankle_angle
    >>> proximal_axis = [[ 0.59, 0.11, 0.16],
    ...                  [-0.13, -0.10, -0.90],
    ...                  [0.94, -0.05, 0.75]]
    >>> distal_axis = [[0.17, 0.69, -0.37],
    ...                [0.14, -0.39, 0.94],
    ...                [-0.16, -0.53, -0.60]]
    >>> np.around(get_ankle_angle(proximal_axis, distal_axis), 2)
    array([0.48, 1.  , 1.56])
    """
    # make inverse matrix of proximal_axis
    inverse_proximal_axis = np.linalg.pinv(proximal_axis)

    # M is the multiplication of the distal_axis and the inverse_proximal_axis
    M = np.matmul(distal_axis, inverse_proximal_axis)

    # This is the angle calculation in YXZ Euler angle
    alpha = np.arctan(M[2][1] / np.sqrt((M[2][0]**2)+(M[2][2]**2)))
    beta = np.arctan(-M[2][0] / M[2][2])
    gamma = np.arctan(-M[0][1] / M[1][1])

    return [alpha, beta, gamma]


def find_joint_center(marker_a, marker_b, marker_c, delta):
    """Calculate the Joint Center function.

    This function is based on physical markers; marker_a,marker_b,marker_c and
    the joint center which will be calulcated in this function are all in the
    same plane.

    Parameters
    ----------
    marker_a, marker_b, marker_c : list
        Three markers x, y, z position of marker_a, marker_b, marker_c.
    delta : float
        The length from marker to joint center, retrieved from subject
        measurement file.

    Returns
    -------
    joint_center : array
        Returns the joint center x, y, z positions in a 1x3 list.

    Examples
    --------
    >>> import numpy as np
    >>> from .static import find_joint_center
    >>> marker_a = [775.41, 788.65, 514.41]
    >>> marker_b = [424.57, 46.17, 305.73]
    >>> marker_c = [618.98, 848.86, 133.52]
    >>> delta = 42.5
    >>> np.around(find_joint_center(marker_a, marker_b, marker_c, delta), 2)
    array([599.66, 843.26,  96.08])
    """
    # make the two vector using 3 markers, which is on the same plane.
    vector_1 = np.subtract(marker_a, marker_c)
    vector_2 = np.subtract(marker_b, marker_c)

    # vector_3 is cross vector of vector_1, vector_2
    # and then it normalized.
    vector_3 = np.cross(vector_1, vector_2)
    vector_3_norm = np.nan_to_num(np.linalg.norm(vector_3))
    if vector_3_norm:
        vector_3 = np.divide(vector_3, vector_3_norm)

    centroid = np.mean([marker_b, marker_c], axis=0)
    length = np.subtract(marker_b, centroid)
    length = np.linalg.norm(length)

    theta = np.arccos(delta/np.linalg.norm(vector_2))
    cosine = np.cos(theta*2)
    sine = np.sin(theta*2)

    ux = vector_3[0]
    uy = vector_3[1]
    uz = vector_3[2]

    # this rotation matrix is called Rodrigues' rotation formula.
    # In order to make a plane, at least 3 number of markers is required which
    # means three physical markers on the segment can make a plane. then the
    # orthogonal vector of the plane will be rotating axis. joint center is
    # determined by rotating the one vector of plane around rotating axis.

    # I'm not sure this is an actual implementation of a Rodrigues' rotation
    # formula.
    rot = np.array([[cosine+ux**2.0*(1.0-cosine), ux*uy*(1.0-cosine)-uz*sine,
                     ux*uz*(1.0-cosine)+uy*sine],
                    [uy*ux*(1.0-cosine)+uz*sine, cosine+uy**2.0*(1.0-cosine),
                     uy*uz*(1.0-cosine)-ux*sine],
                    [uz*ux*(1.0-cosine)-uy*sine, uz*uy*(1.0-cosine)+ux*sine,
                     cosine+uz**2.0*(1.0-cosine)]])

    rotated_axis = rot*(np.matrix(vector_2).transpose())
    rotated_axis = rotated_axis * length/np.linalg.norm(rotated_axis)

    rotated_axis = [rotated_axis[0, 0], rotated_axis[1, 0], rotated_axis[2, 0]]
    joint_center = np.add(rotated_axis, centroid)

    return joint_center
