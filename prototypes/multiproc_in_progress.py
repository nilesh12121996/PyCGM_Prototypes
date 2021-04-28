import numpy as np
import os
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import ctypes
import sys
import functools
import timeit
import inspect
from scipy import sparse

sys.path.append("..")
from utils import pycgmIO

def parameters(*args):
    # decorator parameters are keys
    # finds key in either the marker data or the measurement data
    # sets function parameters accordingly
    def decorator(func):
        params = list(args)

        @functools.wraps(func)
        def find_parameters(self, *args):
            test = func.__getattribute__
            requested_params = [param for param in params]
            return func(self, *requested_params)
        return find_parameters
    return decorator

class pyCGM():

    def __init__(self, measurements, markers):

        # marker data is flat [xyzxyz...] per frame 
        self.measurements = dict(zip(measurements[0], measurements[1]))
        self.marker_data = np.array([np.array(list(marker.values())).flatten() for marker in markers])

        # lists of keys
        self.marker_keys = markers[0].keys()
        self.angle_result_keys = self.angle_result_keys()
        self.axis_result_keys = self.axis_result_keys()

        # mapping dicts
        self.marker_mapping = self.map_marker_slices()
        self.angle_result_mapping = self.map_angle_results()
        self.axis_result_mapping = self.map_axis_results()

        # structured results indexed by [OPTIONAL frame][angle/axis name]
        self.angle_results = self.structure_angle_results(len(self.marker_data))
        self.axis_results = self.structure_axis_results(len(self.marker_data))

        # Structured marker data is indexed by [OPTIONAL frame][marker name]
        # self.marker_data_struct = self.structure_marker_data(self.marker_data)

        # self.results = []


    def angle_result_keys(self):
        return ['Pelvis', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Ankle',
                'L Ankle', 'R Foot', 'L Foot',
                'Head', 'Thorax', 'Neck', 'Spine', 'R Shoulder', 'L Shoulder',
                'R Elbow', 'L Elbow', 'R Wrist', 'L Wrist']

    def map_angle_results(self):
        return {angle_key: index for index, angle_key in enumerate(self.angle_result_keys)}
    

    def axis_result_keys(self):
        return ['Pelvis', 'Hip', 'Knee', 'Ankle', 'Foot', 'Head', 'Thorax', 'RClav',
                'LClav', 'RHum', 'LHum', 'RRad', 'LRad', 'RHand', 'LHand']

    def map_axis_results(self):
        return {axis_key: index for index, axis_key in enumerate(self.axis_result_keys)}

    def structure_angle_results(self, num_frames):
        num_triples = len(self.angle_result_keys)

        angle_row_dtype = np.dtype([(name, '3f4') for name in self.angle_result_keys])
        angle_data_dtype = np.dtype((angle_row_dtype))

        return np.empty((num_frames), dtype=angle_data_dtype)

    def structure_axis_results(self, num_frames):
        num_triples = len(self.axis_result_keys)

        axis_row_dtype = np.dtype([(name, 'f4', (4,4)) for name in self.axis_result_keys])
        axis_data_dtype = np.dtype((axis_row_dtype))

        return np.empty((num_frames), dtype=axis_data_dtype)

    def structure_marker_data(self, marker_frames):

        """
        input: indexed by marker_frames[frame][slice]

            frame     positions
            1       [xyzxyzxyzxyz] 
            2           ...
            3           ...

        output: indexed by structured_marker_data[OPTIONAL frame][MARKERNAME]

            frame     RKNE        LKNE
            1       [x,y,z]     [x,y,z]
            2         ...         ...
            3         ...         ...

        Examples (all equal):

            500th frame, RKNE slice from mapping dict (old)
            marker_frames[500][self.marker_mapping['RKNE']]

            # 500th frame, field named RKNE
            structured_marker_data[500]['RKNE']

            # index 500 of array of ALL RKNEs
            structured_marker_data['RKNE'][500]
        """

        num_triples = len(self.marker_keys)

        if marker_frames.ndim == 1: # if we're working with just 1 frame, add axis so marker_frames[i] is an array and not a coordinate
            marker_frames = marker_frames[np.newaxis]
            num_frames = 1
        else:
            num_frames = len(marker_frames)

        marker_row_dtype = np.dtype([(name, '3f8') for name in self.marker_keys])
        marker_data_dtype = np.dtype((marker_row_dtype))

        structured_marker_data = np.empty((num_frames), dtype=marker_data_dtype)

        for i in range(num_frames):
            structured_marker_data[i] = np.asarray(tuple(np.hsplit(marker_frames[i], num_triples)), dtype=marker_row_dtype)
        
        if structured_marker_data.size == 1:
            return structured_marker_data[0]
        else:
            return structured_marker_data

    def marker_slice(self, key):
        return self.marker_mapping[key]

    def map_marker_slices(self):
        marker_mapping = {marker_key: slice(index*3, index*3+3, 1) for index, marker_key in enumerate(self.marker_keys)}
        return marker_mapping

    def find_in_frame(self, key, frame_index):
        try:
            value = self.structured_marker_data[frame_index][key]
        except ValueError:
            try:
                value = self.measurements[key]
            except KeyError:
                print('Key not found: ', key)
                value = None
        return value

    def multi_run(self):
        flat_rows = self.marker_data

        # num_frames = len(flat_rows)
        # num_coords = len(flat_rows[0])
        # f, c = num_frames, num_coords
        # mp_arr = mp.Array(ctypes.c_double, f*c) # shared, can be used from multiple processes
        # # then in each new process create a new numpy array using:
        # arr = np.frombuffer(mp_arr.get_obj()) # mp_arr and arr share the same memory
        # # make it two-dimensional
        # b = arr.reshape((f,c)) # b and arr share the same memory


        nprocs = os.cpu_count()
        marker_data_blocks = np.vsplit(flat_rows, nprocs)

        processes = []
        frame_index_offset = 0
        for i in range(nprocs):
            processes.append(mp.Process(target=self.run, args=(marker_data_blocks[i], frame_index_offset)))
            frame_index_offset += len(marker_data_blocks[i])
        # processes = [mp.Process(target=self.run, args=(marker_data_blocks[i], len(marker_data_blocks[i]))) for i in range(nprocs)]

        for process in processes:
            process.start()
        for process in processes:
            process.join()
            
        results = []
        for i in range(6000, 1, 500):
            results.append(self.axis_results[i]['Pelvis'])
            print(self.axis_results[i]['Pelvis'])
        print('test')



    def run(self, frames, index_offset):
        for index, frame in enumerate(frames):
            # print('Running frame ', index + index_offset, end='\t')
            self.calc(frame, index + index_offset)

    def calc(self, frame, frame_index):
        marker_struct = self.structure_marker_data(frame)
        self.axis_results[frame_index]['Pelvis'] = self.pelvis_axis(
            marker_struct['RASI'],
            marker_struct['LASI'],
            marker_struct['RPSI'],
            marker_struct['LPSI'],
            marker_struct['RASI']
        )

    def pelvis_axis(self, rasi, lasi, rpsi, lpsi, sacr=None):
        # Get the Pelvis Joint Centre

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

        pycgm_frame_0_value = np.array([[-933.32816389, -4.52677475, 852.69779112, -934.31488037],
                                       [-934.23541808, -3.44793703,  852.8118044, -4.44443512],
                                       [-934.1731894, -4.42988342, 853.82763357, 852.83782959],
                                       [0, 0, 0, 1,]])

        # print('pelvis_axis result ≈ pyCGM/pelvisJointCenter frame 0? ', np.allclose(pelvis, pycgm_frame_0_value))
        return pelvis

class gonzCGM(pyCGM):

    @parameters('RASI', 'LASI', 'RPSI', 'LPSI', 'LeftKneeWidth')
    def pelvis_axis(self, rasi, lasi, rpsi, lpsi, left_knee_width):
        # Get the Pelvis Joint Centre

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

        actual_pycgm_frame_0_value = np.array([[-933.32816389, -4.52677475, 852.69779112, -934.31488037],
                                       [-934.23541808, -3.44793703,  852.8118044, -4.44443512],
                                       [-934.1731894, -4.42988342, 853.82763357, 852.83782959],
                                       [0, 0, 0, 1,]])

        # print('pelvis_axis result ≈ pyCGM/pelvisJointCenter frame 0? ', np.allclose(pelvis, actual_pycgm_frame_0_value))
        return pelvis
    
# measurements = pycgmIO.loadVSK('../SampleData/59993_Frame/59993_Frame_SM.vsk')
# marker_data = pycgmIO.loadData('../SampleData/59993_Frame/59993_Frame_Dynamic.c3d')
measurements = pycgmIO.loadVSK('../SampleData/Sample_2/RoboSM.vsk')
marker_data = pycgmIO.loadData('../SampleData/Sample_2/RoboWalk.c3d')


CGM = pyCGM(measurements, marker_data)
CGM.multi_run()

# customCGM = gonzCGM(measurements, marker_data)
# customCGM.multi_run()

