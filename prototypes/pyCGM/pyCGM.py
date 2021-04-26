import numpy as np
import os
import multiprocessing as mp
import sys
import functools
import csv
import static
from pycgm_calc import CalcAxes, CalcAngles
from utils import pycgmIO


class pyCGM():

    def __init__(self, measurements, static_trial, dynamic_trial):

        # measurements are a dict, marker data is flat [xyzxyz...] per frame
        self.measurements = static.getStatic(static_trial, dict(zip(measurements[0], measurements[1]))) 
        self.marker_data = np.array([np.asarray(list(frame.values())).flatten() for frame in dynamic_trial])

        # map the list of marker names to slices
        self.marker_keys = dynamic_trial[0].keys()
        self.marker_mapping = {marker_key: slice(index*3, index*3+3, 1) for index, marker_key in enumerate(self.marker_keys)}

        # some struct helper attributes
        self.num_frames = len(self.marker_data)
        self.num_axes = len(self.axis_result_keys)
        self.num_axis_floats_per_frame = self.num_axes * 16
        self.axis_results_shape = (self.num_frames, self.num_axes, 4, 4)
        self.num_angles = len(self.angle_result_keys)
        self.num_angle_floats_per_frame = self.num_angles * 3
        self.angle_results_shape = (self.num_frames, self.num_angles, 3)

        # store these for later extension
        self.axis_keys = self.axis_result_keys
        self.angle_keys = self.angle_result_keys

        # map axis indices for angle calcs
        self.axis_mapping = {axis: index for index, axis in enumerate(self.axis_keys)}

        # list of functions and their parameters
        self.current_frame = 0

        # list of functions whose custom parameters have been already set. this allows us to only lookup the @parameters decorator args just one time
        self.funcs_already_known = []

        # add non-overridden default pycgm_calc funcs to funcs list
        self.axis_funcs = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAxes().funcs]
        self.angle_funcs = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAngles().funcs]

        # map function names to indices
        self.axis_func_mapping = {function.__name__: index for index, function in enumerate(self.axis_funcs)}
        self.angle_func_mapping = {function.__name__: index for index, function in enumerate(self.angle_funcs)}

        # default required slices of axis functions
        self.axis_func_parameters = [
                                [
                                    # pelvis_axis 
                                    self.marker_mapping['RASI'],
                                    self.marker_mapping['LASI'],
                                    self.marker_mapping['RPSI'],
                                    self.marker_mapping['LPSI'],
                                    self.marker_mapping['SACR'] if 'SACR' in self.marker_mapping.keys() else None
                                ],

                                [
                                    # hip_axis
                                ],

                                [
                                    # knee_axis
                                ],

                                [
                                    # ankle_axis
                                ],

                                [
                                    # foot_axis
                                ],

                                [
                                    # head_axis
                                ],

                                [
                                    # thorax_axis
                                ],

                                [
                                    # shoulder_axis
                                ],

                                [
                                    # elbow_axis
                                ],

                                [
                                    # wrist_axis
                                ],

                                [
                                    # hand_axis
                                ],
                            ]

        self.angle_func_parameters = [

                                [
                                    # pelvis_angle 
                                    self.measurements['GCS'],
                                    self.axis_mapping['Pelvis']
                                ],

                                [
                                    # hip_angle
                                ],

                                [
                                    # knee_angle
                                ],

                                [
                                    # ankle_angle
                                ],

                                [
                                    # foot_angle
                                ],

                                [
                                    # head_angle
                                ],

                                [
                                    # thorax_angle
                                ],

                                [
                                    # neck_angle
                                ],

                                [
                                    # spine_angle
                                ],

                                [
                                    # shoulder_angle
                                ],

                                [
                                    # elbow_angle
                                ],

                                [
                                    # wrist_angle   
                                ]
                            ]


    def parameters(*args):
        # decorator parameters are keys
        # finds key in either the marker, measurement, or axis results
        # sets function parameters accordingly and calls it properly

        def decorator(func):
            params = list(args)

            @functools.wraps(func)
            def set_required_markers(*args):
                self = args[0]

                try: # find func index, add params to either axis or angle parameter list
                    func_index = self.axis_func_mapping[func.__name__]
                    param_mapping = self.axis_func_parameters
                except KeyError:
                    func_index = self.angle_func_mapping[func.__name__] 
                    param_mapping = self.angle_func_parameters

                if func not in self.funcs_already_known: # only search for parameters if they haven't already been set
                    param_mapping[func_index] = [self.find(param) for param in params]
                    self.funcs_already_known.append(func)

                func_params = [] # call the function properly
                for param in param_mapping[func_index]:
                    if isinstance(param, slice): # marker data slice
                        func_params.append(self.marker_data[self.current_frame][param])
                    elif isinstance(param, float) or isinstance(param, list): # measurement value
                        func_params.append(param)
                    elif isinstance(param, int): # axis mapping index
                        func_params.append(self.current_axes[param])
                    else:
                        func_params.append(None)
                return func(*func_params)
            return set_required_markers
        return decorator

    def add_function(self, name, axes=None, angles=None):
        # add a custom function to pycgm,

        # get func object
        func = getattr(self, name)

        if axes is not None and angles is not None:
            sys.exit('{} must return either an axis or an angle, not both'.format(func))
        if axes is None and angles is None:
            sys.exit('{} must return a custom axis or angle. if the axis or angle already exists by default, just extend by using the @pyCGM.parameters decorator'.format(func))

        # append to func list, update mapping, add empty parameters list for parameters decorator to append to
        if axes is not None: # extend axes and update
            self.axis_funcs.append(func)
            self.axis_func_parameters.append([])
            self.axis_func_mapping = {function.__name__: index for index, function in enumerate(self.axis_funcs)}
            self.axis_keys.extend(axes)
            self.num_axes = len(self.axis_keys)
            self.axis_mapping = {axis: index for index, axis in enumerate(self.axis_keys)}
            self.num_axis_floats_per_frame = self.num_axes * 16
            self.axis_results_shape = (self.num_frames, self.num_axes, 4, 4)

        if angles is not None: # extend angles and update 
            self.angle_funcs.append(func)
            self.angle_func_mapping = {function.__name__: index for index, function in enumerate(self.angle_funcs)}
            self.angle_keys.extend(angles)
            self.num_angles = len(self.axis_keys)
            self.num_angle_floats_per_frame = self.num_axes * 3
            self.angle_results_shape = (self.num_frames, self.num_angles, 3)

    @property
    def angle_result_keys(self):
        # list of default angle result names

        return ['Pelvis', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Ankle',
                'L Ankle', 'R Foot', 'L Foot',
                'Head', 'Thorax', 'Neck', 'Spine', 'R Shoulder', 'L Shoulder',
                'R Elbow', 'L Elbow', 'R Wrist', 'L Wrist']

    @property
    def axis_result_keys(self):
        # list of default axis result names

        return ['Pelvis', 'Hip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head',
                'Thorax', 'RClav', 'LClav', 'RHum', 'LHum', 'RRad', 'LRad', 'RHand', 'LHand']

    def find(self, key):
        value = None
        for mapping in [self.marker_mapping, self.measurements, self.axis_mapping]:
            try:
                value = mapping[key] 
            except KeyError:
                continue
        return value


    def check_robo_results_accuracy(self, axis_results):
        # test unstructured pelvis axes against existing csv output file

        actual_results = np.genfromtxt('SampleData/Sample_2/RoboResults_pycgm.csv', delimiter=',')
        pelvis_OXYZ = [row[58:70] for row in actual_results]
        accurate = True
        for i, frame in enumerate(pelvis_OXYZ):
            if not np.allclose(frame[:3], axis_results[i][0][:3,3]):
                accurate = False
            if not np.allclose(frame[3:6], axis_results[i][0][0, :3]):
                accurate = False
            if not np.allclose(frame[6:9], axis_results[i][0][1, :3]):
                accurate = False
            if not np.allclose(frame[9:12], axis_results[i][0][2, :3]):
                accurate = False
        print('All pelvis results in line with RoboResults_pycgm.csv: ', accurate)

    def structure_trial_axes(self, axis_results):
        # takes a flat array of floats that represent the 4x4 axes at each frame
        # returns a structured array, indexed by result[optional frame slice or index][axis name]

        # uncomment to test against robo results pelvis
        # check_robo_results_accuracy(axis_results.reshape(self.axis_results_shape))

        axis_row_dtype = np.dtype([(name, 'f4', (4, 4)) for name in self.axis_keys])
        return np.array([tuple(frame) for frame in axis_results.reshape(self.axis_results_shape)], dtype=axis_row_dtype)

    def structure_trial_angles(self, angle_results):
        # takes a flat array of floats that represent the 3x1 angles at each frame
        # returns a structured array, indexed by result[optional frame slice or index][angle name]

        angle_row_dtype = np.dtype([(name, 'f4', (3,)) for name in self.angle_keys])
        return np.array([tuple(frame) for frame in angle_results.reshape(self.angle_results_shape)], dtype=angle_row_dtype)

    def multi_run(self, cores=None):
        # parallelize on blocks of frames 

        flat_rows = self.marker_data

        # create a shared array to store axis and angle results
        shared_axes = mp.RawArray('f', self.num_frames * self.num_axes * 16)
        shared_angles = mp.RawArray('f', self.num_frames * self.num_angles * 3)
        nprocs = cores if cores is not None else os.cpu_count() - 1
        marker_data_blocks = np.array_split(flat_rows, nprocs)

        processes = []
        frame_index_offset = 0
        for i in range(nprocs):
            frame_count = len(marker_data_blocks[i])

            processes.append(mp.Process(target=self.run,
                                        args=(marker_data_blocks[i],
                                              frame_index_offset,
                                              frame_index_offset + frame_count,
                                              self.num_axis_floats_per_frame,
                                              self.num_angle_floats_per_frame,
                                              shared_axes,
                                              shared_angles)))

            frame_index_offset += frame_count

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # structure flat axis and angle results
        self.axes = self.structure_trial_axes(np.frombuffer(shared_axes, dtype=np.float32))
        self.angles = self.structure_trial_angles(np.frombuffer(shared_angles, dtype=np.float32))

    def run(self, frames=None, index_offset=None, index_end=None, axis_results_size=None, angle_results_size=None, shared_axes=None, shared_angles=None):

        flat_axis_results = np.array([], dtype=np.float32)
        flat_angle_results = np.array([], dtype=np.float32)

        if shared_angles is not None:  # multiprocessing, write to shared memory
            shared_axes = np.frombuffer(shared_axes, dtype=np.float32)
            shared_angles = np.frombuffer(shared_angles, dtype=np.float32)
            self.current_frame = index_offset # for the decorator

            for frame in frames:
                flat_axis_results = np.append(flat_axis_results, self.calc(frame)[0].flatten())
                flat_angle_results = np.append(flat_angle_results, self.calc(frame)[1].flatten())
                self.current_frame += 1 # for the decorator


            shared_axes[index_offset * axis_results_size: index_end * axis_results_size] = flat_axis_results
            shared_angles[index_offset * angle_results_size: index_end * angle_results_size] = flat_angle_results

        else:  # single core, just calculate and structure
            for frame in self.marker_data:
                flat_axis_results = np.append(flat_axis_results, self.calc(frame)[0].flatten())
                flat_angle_results = np.append(flat_angle_results, self.calc(frame)[1].flatten())
                self.current_frame += 1 # for the decorator 

            # structure flat axis and angle results
            self.axes = self.structure_trial_axes(flat_axis_results)
            self.angles = self.structure_trial_angles(flat_angle_results)

    def calc(self, frame):
        axis_results = []
        angle_results = []

        for index, func in enumerate(self.axis_funcs): # calculate angles
            # old way: ret_axes = func(*[frame[req_slice] if isinstance(req_slice, slice) else req_slice for req_slice in self.axis_func_parameters[index]])
            # ^ did not account for axis or angle calcs requiring other axes

            axis_params = []
            for param in self.axis_func_parameters[index]:
                if isinstance(param, slice): # marker data slice
                    axis_params.append(frame[param])
                elif isinstance(param, float) or isinstance(param, list): # measurement value
                    axis_params.append(param)
                elif isinstance(param, int): # axis mapping index
                    axis_params.append(axis_results[param])
                else:
                    axis_params.append(None)

            ret_axes = func(*axis_params)

            if ret_axes.ndim == 3: # multiple axes returned by one function
                for axis in ret_axes:
                    axis_results.append(axis)
            else:
                axis_results.append(ret_axes)

            self.current_axes = axis_results # for the decorator

        for index, func in enumerate(self.angle_funcs): # calculate angles
            angle_params = []
            for param in self.angle_func_parameters[index]:
                if isinstance(param, slice): # marker data slice
                    angle_params.append(frame[param])
                elif isinstance(param, float) or isinstance(param, list): # measurement value
                    angle_params.append(param)
                elif isinstance(param, int): # axis mapping index
                    angle_params.append(axis_results[param])
                else:
                    angle_params.append(None)

            ret_angles = func(*angle_params)

            if ret_angles.ndim == 2: # multiple angles returned by one function
                for angle in ret_angles:
                    angle_results.append(angle)
            else:
                angle_results.append(ret_angles)
            
        return [np.asarray(axis_results), np.asarray(angle_results)]
