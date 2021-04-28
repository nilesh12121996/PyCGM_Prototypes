import numpy as np
import sys

sys.path.append("..")
from utils import pycgmIO

class pyCGM():

    def __init__(self, measurements, markers):

        # marker data is flat [xyzxyz...] per frame 
        self.measurements = measurements
        self.marker_data = np.array([np.array(list(marker.values())).flatten() for marker in markers])

        # lists of keys
        self.marker_keys = markers[0].keys()
        self.measurement_keys = measurements[0]
        self.angle_result_keys = self.get_angle_result_keys()
        self.axis_result_keys = self.get_axis_result_keys()

        # mapping dicts
        self.marker_mapping = self.map_marker_slices()
        self.measurement_mapping = self.map_measurements()
        self.angle_result_mapping = self.map_angle_results()
        self.axis_result_mapping = self.map_axis_results()


        # structured_measurement_data indexed by structured_marker_data[measurement name]
        self.structured_measurement_data = self.structure_measurement_data(self.measurements)

        # structured results indexed by [OPTIONAL frame][angle/axis name]
        self.structured_angle_results = self.structure_angle_results(len(self.marker_data))
        self.structured_axis_results = self.structure_axis_results(len(self.marker_data))

        # Structured marker data is indexed by [OPTIONAL frame][marker name]
        # self.structured_marker_data = self.structure_marker_data(self.marker_data)

        # self.results = []
        self.multi_run(self.marker_data)


    def get_angle_result_keys(self):
        return ['Pelvis', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Ankle',
                'L Ankle', 'R Foot', 'L Foot',
                'Head', 'Thorax', 'Neck', 'Spine', 'R Shoulder', 'L Shoulder',
                'R Elbow', 'L Elbow', 'R Wrist', 'L Wrist']

    def map_angle_results(self):
        return {angle_key: index for index, angle_key in enumerate(self.angle_result_keys)}
    

    def get_axis_result_keys(self):
        return ['PELO', 'PELX', 'PELY', 'PELZ', 'HIPO', 'HIPX', 'HIPY', 'HIPZ', 'R KNEO',
                'R KNEX', 'R KNEY', 'R KNEZ', 'L KNEO', 'L KNEX', 'L KNEY', 'L KNEZ', 'R ANKO',
                'R ANKX', 'R ANKY', 'R ANKZ', 'L ANKO', 'L ANKX', 'L ANKY', 'L ANKZ', 'R FOOO',
                'R FOOX', 'R FOOY', 'R FOOZ', 'L FOOO', 'L FOOX', 'L FOOY', 'L FOOZ', 'HEAO',
                'HEAX', 'HEAY', 'HEAZ', 'THOO', 'THOX', 'THOY', 'THOZ', 'R CLAO', 'R CLAX',
                'R CLAY', 'R CLAZ', 'L CLAO', 'L CLAX', 'L CLAY', 'L CLAZ', 'R HUMO', 'R HUMX',
                'R HUMY', 'R HUMZ', 'L HUMO', 'L HUMX', 'L HUMY', 'L HUMZ', 'R RADO', 'R RADX',
                'R RADY', 'R RADZ', 'L RADO', 'L RADX', 'L RADY', 'L RADZ', 'R HANO', 'R HANX',
                'R HANY', 'R HANZ', 'L HANO', 'L HANX', 'L HANY', 'L HANZ']

    def map_axis_results(self):
        return {axis_key: index for index, axis_key in enumerate(self.axis_result_keys)}


    def structure_measurement_data(self, measurements):
        measurement_array_dtype = np.dtype([(name, 'f8') for name in self.measurement_keys])
        return np.asarray(tuple((measurements[1])), dtype=measurement_array_dtype)
    
    def structure_angle_results(self, num_frames):
        num_triples = len(self.angle_result_keys)

        angle_row_dtype = np.dtype([(name, 'f8', (4,4)) for name in self.angle_result_keys])
        angle_data_dtype = np.dtype((angle_row_dtype))

        return np.empty((num_frames), dtype=angle_data_dtype)

    def structure_axis_results(self, num_frames):
        num_triples = len(self.axis_result_keys)

        axis_row_dtype = np.dtype([(name, '3f8') for name in self.axis_result_keys])
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

    def map_measurements(self):
        measurement_mapping = {measurement_key: index for index, measurement_key in enumerate(self.measurement_keys)}
        return measurement_mapping

    def multi_run(self, marker_data):
        # TODO split between cores
        self.run(marker_data[:3])

    def run(self, frames):
        for index, frame in enumerate(frames):
            print('Running frame ', index, end='\t')
            self.calc(frame, index)
    

    def calc(self, frame, frame_index):
        marker_struct = self.structure_marker_data(frame)
        # TODO, this is actually an axis and NOT an angle
        # iron out the format of the results, might just have to flip them
        self.structured_angle_results[frame_index]['Pelvis'] = (pyCGM.pelvis_axis(
            marker_struct['RASI'],
            marker_struct['LASI'],
            marker_struct['RPSI'],
            marker_struct['LPSI'],
            marker_struct['RASI'],
        ))

    def pelvis_axis( rasi: np.ndarray, lasi: np.ndarray, rpsi: np.ndarray, lpsi: np.ndarray, sacr: np.ndarray):
        # Get the Pelvis Joint Centre
        # this is David's implementation, but i deleted his beautiful docstring to make it compact (sorry David)

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

        print('pelvis_axis result ≈ pyCGM/pelvisJointCenter frame 0? ', np.allclose(pelvis, actual_pycgm_frame_0_value))
        return pelvis
    
# measurements = pycgmIO.loadVSK('../SampleData/59993_Frame/59993_Frame_SM.vsk')
# marker_data = pycgmIO.loadData('../SampleData/59993_Frame/59993_Frame_Dynamic.c3d')
measurements = pycgmIO.loadVSK('../SampleData/Sample_2/RoboSM.vsk')
marker_data = pycgmIO.loadData('../SampleData/Sample_2/RoboWalk.c3d')


CGM = pyCGM(measurements, marker_data)

