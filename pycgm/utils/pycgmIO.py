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
    from . import c3dpy3 as c3d
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


def writeKinetics(CoM_output, kinetics):
    """Uses numpy.save to write kinetics data as an .npy file.

    Parameters
    ----------
    CoM_output : file, str, or Path
        Full path of the file to be saved to or a file object
        or a filename.
    kinetics : array_like
        Array data to be saved.

    Examples
    --------
    >>> import tempfile
    >>> tmpdirName = tempfile.mkdtemp()
    >>> from numpy import load
    >>> import os
    >>> from shutil import rmtree
    >>> CoM_output = os.path.join(tmpdirName, 'CoM')
    >>> kinetics = [[246.57466721, 313.55662383, 1026.56323492],
    ...             [246.59137623, 313.6216639, 1026.56440096],
    ...             [246.60850798, 313.6856272, 1026.56531282]]
    >>> writeKinetics(CoM_output, kinetics)
    >>> load(CoM_output + '.npy') #doctest: +NORMALIZE_WHITESPACE
    array([[ 246.57466721, 313.55662383, 1026.56323492],
           [ 246.59137623, 313.6216639 , 1026.56440096],
           [ 246.60850798, 313.6856272 , 1026.56531282]])
    >>> rmtree(tmpdirName)
    """
    np.save(CoM_output, kinetics)


def writeResult(data, filename, **kargs):
    """Writes the result of the calculation into a csv file.

    Lines 0-6 of the output csv are headers. Lines 7 and onwards
    are angle or axis calculations for each frame. For example,
    line 7 of the csv is output for frame 0 of the motion capture.
    The first element of each row of ouput is the frame number.

    Parameters
    ----------
    data : array_like
        Motion capture data as a matrix of frames as rows.
        Each row is a numpy array of length 273.
        Indices 0-56 correspond to the values for angles.
        Indices 57-272 correspond to the values for axes.
        See Examples.
    filename : str
        Full path of the csv to be saved. Do not include '.csv'.
    **kargs : dict
        Dictionary of keyword arguments as follows.

        delimiter : str, optional
            String to be used as the delimiter for the csv file. The
            default delimiter is ','.
        angles : bool or array, optional
            True or false to save angles, or a list of angles to save.
            True by default.
        axis : bool or array, optional
            True or false to save axis, or a list of axis to save.
            True by default.

    Examples
    --------
    >>> from numpy import array, zeros
    >>> import os
    >>> from shutil import rmtree
    >>> import tempfile
    >>> tmpdirName = tempfile.mkdtemp()

    Prepare a frame of data to write to csv. This example writes joint angle values
    for the first joint, the pelvis, and axis values for the pelvis origin, PELO.

    >>> frame = zeros(273)
    >>> angles = array([-0.308494914509454, -6.12129279337001, 7.57143110215171])
    >>> for i in range(len(angles)):
    ...     frame[i] = angles[i]
    >>> axis = array([-934.314880371094, -4.44443511962891, 852.837829589844])
    >>> for i in range(len(axis)):
    ...     frame[i+57] = axis[i]
    >>> data = [frame]
    >>> outfile = os.path.join(tmpdirName, 'output')

    Writing angles only.

    >>> writeResult(data, outfile, angles=True, axis=False)
    >>> with open(outfile + '.csv') as file:
    ...     lines = file.readlines()
    >>> result = lines[7].strip().split(',')
    >>> result #doctest: +NORMALIZE_WHITESPACE
    ['0.000000000000000',
     '-0.308494914509454', '-6.121292793370010', '7.571431102151710',...]

    Writing axis only.

    >>> writeResult(data, outfile, angles=False, axis=True)
    (1, 273)...
    >>> with open(outfile + '.csv') as file:
    ...     lines = file.readlines()
    >>> result = lines[7].strip().split(',')
    >>> result #doctest: +NORMALIZE_WHITESPACE
    ['0.000000000000000',
     '-934.314880371093977', '-4.444435119628910', '852.837829589843977',...]
    >>> rmtree(tmpdirName)
    """
    labelsAngs = ['Pelvis', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Ankle',
                            'L Ankle', 'R Foot', 'L Foot',
                            'Head', 'Thorax', 'Neck', 'Spine', 'R Shoulder', 'L Shoulder',
                            'R Elbow', 'L Elbow', 'R Wrist', 'L Wrist']

    labelsAxis = ["PELO", "PELX", "PELY", "PELZ", "HIPO", "HIPX", "HIPY", "HIPZ",
                  "R KNEO", "R KNEX", "R KNEY", "R KNEZ", "L KNEO", "L KNEX", "L KNEY", "L KNEZ",
                  "R ANKO", "R ANKX", "R ANKY", "R ANKZ", "L ANKO", "L ANKX", "L ANKY", "L ANKZ",
                  "R FOOO", "R FOOX", "R FOOY", "R FOOZ", "L FOOO", "L FOOX", "L FOOY", "L FOOZ",
                  "HEAO", "HEAX", "HEAY", "HEAZ", "THOO", "THOX", "THOY", "THOZ", "R CLAO", "R CLAX",
                  "R CLAY", "R CLAZ", "L CLAO", "L CLAX", "L CLAY", "L CLAZ", "R HUMO", "R HUMX",
                  "R HUMY", "R HUMZ", "L HUMO", "L HUMX", "L HUMY", "L HUMZ", "R RADO", "R RADX",
                  "R RADY", "R RADZ", "L RADO", "L RADX", "L RADY", "L RADZ", "R HANO", "R HANX",
                  "R HANY", "R HANZ", "L HANO", "L HANX", "L HANY", "L HANZ"]

    outputAngs = True
    outputAxis = True
    dataFilter = None
    delimiter = ","
    filterData = []
    if 'delimiter' in kargs:
        delimiter = kargs['delimiter']
    if 'angles' in kargs:
        if kargs['angles'] == True:
            outputAngs = True
        elif kargs['angles'] == False:
            outputAngs = False
            labelsAngs = []
        elif isinstance(kargs['angles'], (list, tuple)):
            filterData = [
                i*3 for i in range(len(labelsAngs)) if labelsAngs[i] not in kargs['angles']]
            if len(filterData) == 0:
                outputAngs = False
            labelsAngs = [i for i in labelsAngs if i in kargs['angles']]

    if 'axis' in kargs:
        if kargs['axis'] == True:
            outputAxis = True
        elif kargs['axis'] == False:
            outputAxis = False
            labelsAxis = []
        elif isinstance(kargs['axis'], (list, tuple)):
            filteraxis = [
                i*3 for i in range(len(labelsAxis)) if labelsAxis[i] not in kargs['axis']]
            filterData = filterData+filteraxis
            if len(filteraxis) == 0:
                outputAxis = False
            labelsAxis = [i for i in labelsAxis if i in kargs['axis']]

    if len(filterData) > 0:
        filterData = np.repeat(filterData, 3)
        filterData[1::3] = filterData[1::3]+1
        filterData[2::3] = filterData[2::3]+2

    if outputAngs == outputAxis == False:
        return
    elif outputAngs == False:
        print(np.shape(data))
        dataFilter = np.transpose(data)
        dataFilter = dataFilter[57:273]
        dataFilter = np.transpose(dataFilter)
        print(np.shape(dataFilter))
        print(filterData)
        filterData = [i for i in filterData]
        print(filterData)
    elif outputAxis == False:
        dataFilter = np.transpose(data)
        dataFilter = dataFilter[0:57]
        dataFilter = np.transpose(dataFilter)

    if len(filterData) > 0:
        if type(dataFilter) == type(None):
            dataFilter = np.delete(data, filterData, 1)
        else:
            dataFilter = np.delete(dataFilter, filterData, 1)
    if type(dataFilter) == type(None):
        dataFilter = data
    header = ","
    headerAngs = ["Joint Angle,,,", ",,,x = flexion/extension angle",
                  ",,,y= abudction/adduction angle", ",,,z = external/internal rotation angle", ",,,"]
    headerAxis = ["Joint Coordinate", ",,,###O = Origin", ",,,###X = X axis orientation",
                  ",,,###Y = Y axis orientation", ",,,###Z = Z axis orientation"]
    for angs, axis in zip(headerAngs, headerAxis):
        if outputAngs == True:
            header = header+angs+",,,"*(len(labelsAngs)-1)
        if outputAxis == True:
            header = header+axis+",,,"*(len(labelsAxis)-1)
        header = header+"\n"
    labels = ","
    if len(labelsAngs) > 0:
        labels = labels+",,,".join(labelsAngs)+",,,"
    if len(labelsAxis) > 0:
        labels = labels+",,,".join(labelsAxis)
    labels = labels+"\n"
    if pyver == 2:
        xyz = "frame num,"+"X,Y,Z,"*(len(dataFilter[0])/3)
    else:
        xyz = "frame num,"+"X,Y,Z,"*(len(dataFilter[0])//3)
    header = header+labels+xyz
    # Creates the frame numbers
    frames = np.arange(len(dataFilter), dtype=dataFilter[0].dtype)
    # Put the frame numbers in the first dimension of the data
    dataFilter = np.column_stack((frames, dataFilter))
    start = 1500
    end = 3600
    np.savetxt(filename+'.csv', dataFilter,
               delimiter=delimiter, header=header, fmt="%.15f")
