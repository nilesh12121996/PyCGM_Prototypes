"""decorator approach"""

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

def partial(func, *part_args):
    def wrapper(*extra_args):
        markers = list(part_args)
        markers.extend(extra_args)
        return func(*markers)

    return wrapper


class Gen_PJC_Func:
    def __init__(self, frame):
        self.frame = frame
        self._pelvis_axis = None
    
    def pelvisJointCenter(frame):
        """Make the Pelvis Axis.
        Takes in a dictionary of x,y,z positions and marker names, as well as an index
        Calculates the pelvis joint center and axis and returns both.
        Markers used: RASI, LASI, RPSI, LPSI
        Other landmarks used: origin, sacrum
        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure [1]_ and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: Cross product of x_axis and y_axis.
        Parameters
        ----------
        frame : dict
            Dictionaries of marker lists.
        Returns
        -------
        pelvis : array
            Returns an array that contains the pelvis origin in a 1x3 array of xyz values,
            which is then followed by a [1x3, 3x3, 1x3] array composed of the 
            pelvis x, y, z axis components, and the sacrum x,y,z position.
        References
        ----------
        .. [1] M. P. Kadaba, H. K. Ramakrishnan, and M. E. Wootten, “Measurement of
                lower extremity kinematics during level walking,” J. Orthop. Res.,
                vol. 8, no. 3, pp. 383–392, May 1990, doi: 10.1002/jor.1100080310.
        Examples
        --------
        >>> import numpy as np
        >>> from .pyCGM import pelvisJointCenter
        >>> frame = {'RASI': np.array([ 395.36,  428.09, 1036.82]),
        ...          'LASI': np.array([ 183.18,  422.78, 1033.07]),
        ...          'RPSI': np.array([ 341.41,  246.72, 1055.99]),
        ...          'LPSI': np.array([ 255.79,  241.42, 1057.30]) }
        >>> [arr.round(2) for arr in pelvisJointCenter(frame)] #doctest: +NORMALIZE_WHITESPACE
        [array([ 289.27,  425.43, 1034.94]), array([[ 289.25,  426.43, 1034.83],
        [ 288.27,  425.41, 1034.93],
        [ 289.25,  425.55, 1035.94]]), array([ 298.6 ,  244.07, 1056.64])]
        >>> frame = {'RASI': np.array([ 395.36,  428.09, 1036.82]),
        ...          'LASI': np.array([ 183.18,  422.78, 1033.07]),
        ...          'SACR': np.array([ 294.60,  242.07, 1049.64]) }
        >>> [arr.round(2) for arr in pelvisJointCenter(frame)] #doctest: +NORMALIZE_WHITESPACE
        [array([ 289.27,  425.43, 1034.94]), array([[ 289.25,  426.43, 1034.87],
        [ 288.27,  425.41, 1034.93],
        [ 289.25,  425.51, 1035.94]]), array([ 294.6 ,  242.07, 1049.64])]
        """
        # Get the Pelvis Joint Centre

        #REQUIRED MARKERS:
        # RASI
        # LASI
        # RPSI
        # LPSI

        RASI = frame['RASI']
        LASI = frame['LASI']

        try:
            RPSI = frame['RPSI']
            LPSI = frame['LPSI']
            #  If no sacrum, mean of posterior markers is used as the sacrum
            sacrum = (RPSI+LPSI)/2.0
        except:
            pass #going to use sacrum marker

        #  If no sacrum, mean of posterior markers is used as the sacrum
        if 'SACR' in frame:
            sacrum = frame['SACR']

        # REQUIRED LANDMARKS:
        # origin
        # sacrum

        # Origin is Midpoint between RASI and LASI
        origin = (RASI+LASI)/2.0

        # This calculate the each axis
        # beta1,2,3 is arbitrary name to help calculate.
        beta1 = origin-sacrum
        beta2 = LASI-RASI

        # Y_axis is normalized beta2
        y_axis = beta2/norm3d(beta2)

        # X_axis computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990)
        # and then normalized.
        beta3_cal = np.dot(beta1,y_axis)
        beta3_cal2 = beta3_cal*y_axis
        beta3 = beta1-beta3_cal2
        x_axis = beta3/norm3d(beta3)

        # Z-axis is cross product of x_axis and y_axis.
        z_axis = cross(x_axis,y_axis)

        # Add the origin back to the vector
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin

        pelvis_axis = np.asarray([x_axis,y_axis,z_axis])

        pelvis = [origin,pelvis_axis,sacrum] #probably don't need to return sacrum

        return pelvis

    def run(self):
        # First Calculate Pelvis
        self._pelvis_axis = self.pelvisJointCenter()

    @property
    def pelvis_axis(self):
        if self._pelvis_axis is None:
            self.run()
        return self._pelvis_axis


class Mod_PJC_Func(CGM):
    def pelvisJointCenter(frame):
        """Make the Pelvis Axis.
        Takes in a dictionary of x,y,z positions and marker names, as well as an index
        Calculates the pelvis joint center and axis and returns both.
        Markers used: RASI, LASI, RPSI, LPSI
        Other landmarks used: origin, sacrum
        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure [1]_ and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: Cross product of x_axis and y_axis.
        Parameters
        ----------
        frame : dict
            Dictionaries of marker lists.
        Returns
        -------
        pelvis : array
            Returns an array that contains the pelvis origin in a 1x3 array of xyz values,
            which is then followed by a [1x3, 3x3, 1x3] array composed of the 
            pelvis x, y, z axis components, and the sacrum x,y,z position.
        References
        ----------
        .. [1] M. P. Kadaba, H. K. Ramakrishnan, and M. E. Wootten, “Measurement of
                lower extremity kinematics during level walking,” J. Orthop. Res.,
                vol. 8, no. 3, pp. 383–392, May 1990, doi: 10.1002/jor.1100080310.
        Examples
        --------
        >>> import numpy as np
        >>> from .pyCGM import pelvisJointCenter
        >>> frame = {'RASI': np.array([ 395.36,  428.09, 1036.82]),
        ...          'LASI': np.array([ 183.18,  422.78, 1033.07]),
        ...          'RPSI': np.array([ 341.41,  246.72, 1055.99]),
        ...          'LPSI': np.array([ 255.79,  241.42, 1057.30]) }
        >>> [arr.round(2) for arr in pelvisJointCenter(frame)] #doctest: +NORMALIZE_WHITESPACE
        [array([ 289.27,  425.43, 1034.94]), array([[ 289.25,  426.43, 1034.83],
        [ 288.27,  425.41, 1034.93],
        [ 289.25,  425.55, 1035.94]]), array([ 298.6 ,  244.07, 1056.64])]
        >>> frame = {'RASI': np.array([ 395.36,  428.09, 1036.82]),
        ...          'LASI': np.array([ 183.18,  422.78, 1033.07]),
        ...          'SACR': np.array([ 294.60,  242.07, 1049.64]) }
        >>> [arr.round(2) for arr in pelvisJointCenter(frame)] #doctest: +NORMALIZE_WHITESPACE
        [array([ 289.27,  425.43, 1034.94]), array([[ 289.25,  426.43, 1034.87],
        [ 288.27,  425.41, 1034.93],
        [ 289.25,  425.51, 1035.94]]), array([ 294.6 ,  242.07, 1049.64])]
        """
        # Get the Pelvis Joint Centre

        #REQUIRED MARKERS:
        # RASI
        # LASI
        # RPSI
        # LPSI

        RASI = frame['RASI']
        LASI = frame['LASI']

        try:
            RPSI = frame['RPSI']
            LPSI = frame['LPSI']
            #  If no sacrum, mean of posterior markers is used as the sacrum
            sacrum = (RPSI+LPSI)/2.0
        except:
            pass #going to use sacrum marker

        #  If no sacrum, mean of posterior markers is used as the sacrum
        if 'SACR' in frame:
            sacrum = frame['SACR']

        # REQUIRED LANDMARKS:
        # origin
        # sacrum

        # Origin is Midpoint between RASI and LASI
        origin = (RASI+LASI)/2.0

        # This calculate the each axis
        # beta1,2,3 is arbitrary name to help calculate.
        beta1 = origin-sacrum
        beta2 = LASI-RASI

        # Y_axis is normalized beta2
        y_axis = beta2/norm3d(beta2)

        # X_axis computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990)
        # and then normalized.
        beta3_cal = np.dot(beta1,y_axis)
        beta3_cal2 = beta3_cal*y_axis
        beta3 = beta1-beta3_cal2
        x_axis = beta3/norm3d(beta3)

        # Z-axis is cross product of x_axis and y_axis.
        z_axis = cross(x_axis,y_axis)

        # Add the origin back to the vector
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin

        pelvis_axis = np.asarray([x_axis,y_axis,z_axis])

        pelvis = [origin,pelvis_axis,sacrum] #probably don't need to return sacrum

        return pelvis