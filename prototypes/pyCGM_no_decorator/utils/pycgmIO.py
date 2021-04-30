"""
This file is used for the input and output of pycgm functions.
"""

# Copyright (c) 2015 Mathew Schwartz <umcadop@gmail.com>

import sys
if sys.version_info[0] == 2:
    import c3d
    pyver = 2
    print("Using python 2 c3d loader")

else:
    import c3dpy3 as c3d
    pyver = 3
    print("Using python 3 c3d loader - c3dpy3")

try:
    from ezc3d import c3d as ezc
    useEZC3D = True
    print("EZC3D Found, using instead of Python c3d")
except:
    useEZC3D = False

from math import *
import numpy as np
import xml.etree.ElementTree as ET
import os
import errno


def loadData(filename, rawData=True):
    """Loads motion capture data from a c3d file.

    Parameters
    ----------
    filename : str
        Path of the c3d file to be loaded.

    Returns
    -------
    data : array
        `data` is a list of dict. Each dict represents one frame in
        the trial.

    Examples
    --------
    RoboResults.c3d in SampleData are used to
    test the output.

    >>> c3dFile = 'SampleData/Sample_2/RoboStatic.c3d'
    >>> c3dData = loadData(c3dFile)
    SampleData/Sample_2/RoboStatic.c3d

    Testing for some values from the loaded c3d file.

    >>> c3dData[0]['RHNO'] #doctest: +NORMALIZE_WHITESPACE
    array([-259.45016479, -844.99560547, 1464.26330566])
    >>> c3dData[0]['C7'] #doctest: +NORMALIZE_WHITESPACE
    array([-2.20681717e+02, -1.07236075e+00, 1.45551550e+03])
    """
    print(filename)

    reader = c3d.Reader(open(filename, 'rb'))
    labels = reader.get('POINT:LABELS').string_array
    data = []
    dataunlabeled = []

    markers = [str(label.rstrip()) for label in labels]

    for frame_no, points, analog in reader.read_frames(True, True):
        data_dict = {}
        data_unlabeled = {}
        for label, point in zip(markers, points):
            # Create a dictionary with format LFHDX: 123
            if label[0] == '*':
                if point[0] != np.nan:
                    data_unlabeled[label] = point
            else:
                data_dict[label] = point

        data.append(data_dict)
        dataunlabeled.append(data_unlabeled)

    # add any missing keys
    keys = ['RASI', 'LASI', 'RPSI', 'LPSI', 'RTHI', 'LTHI', 'RKNE', 'LKNE', 'RTIB',
            'LTIB', 'RANK', 'LANK', 'RTOE', 'LTOE', 'LFHD', 'RFHD', 'LBHD', 'RBHD',
            'RHEE', 'LHEE', 'CLAV', 'C7', 'STRN', 'T10', 'RSHO', 'LSHO', 'RELB', 'LELB',
            'RWRA', 'RWRB', 'LWRA', 'LWRB', 'RFIN', 'LFIN']
    for frame in data:
        for key in keys:
            frame.setdefault(key, [np.nan, np.nan, np.nan])
    return data


def loadVSK(filename, dict=True):
    """Open and load a vsk file.

    Parameters
    ----------
    filename : str
        Path to the vsk file to be loaded
    dict : bool, optional
        Returns loaded vsk file values as a dictionary if False.
        Otherwise, return as an array.

    Returns
    -------
    [vsk_keys, vsk_data] : array
        `vsk_keys` is a list of labels. `vsk_data` is a list of values
        corresponding to the labels in `vsk_keys`.

    Examples
    --------
    RoboSM.vsk in SampleData is used to test the output.

    >>> filename = 'SampleData/Sample_2/RoboSM.vsk'
    >>> result = loadVSK(filename)
    >>> vsk_keys = result[0]
    >>> vsk_data = result[1]
    >>> vsk_keys
    ['Bodymass', 'Height', 'InterAsisDistance',...]
    >>> vsk_data
    [72.0, 1730.0, 281.118011474609,...]

    Return as a dictionary.

    >>> result = loadVSK(filename, False)
    >>> type(result)
    <...'dict'>

    Testing for some dictionary values.

    >>> result['Bodymass']
    72.0
    >>> result['RightStaticPlantFlex']
    0.17637075483799
    """
    # Check if the filename is valid
    # if not, return None
    if filename == '':
        return None

    # Create an XML tree from file
    tree = ET.parse(filename)
    # Get the root of the file
    # <KinematicModel>
    root = tree.getroot()

    # Store the values of each parameter in a dictionary
    # the format is (NAME,VALUE)
    vsk_keys = [r.get('NAME') for r in root[0]]
    vsk_data = []
    for R in root[0]:
        val = (R.get('VALUE'))
        if val == None:
            val = 0
        vsk_data.append(float(val))

    # print vsk_keys
    if dict == False:
        vsk = {}
        for key, data in zip(vsk_keys, vsk_data):
            vsk[key] = data
        return vsk

    return [vsk_keys, vsk_data]
