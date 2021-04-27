import numpy as np
from itertools import chain
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
        self.num_axes = len(self.default_axis_keys)
        self.num_axis_floats_per_frame = self.num_axes * 16
        self.axis_results_shape = (self.num_frames, self.num_axes, 4, 4)
        self.num_angles = len(self.default_angle_keys)
        self.num_angle_floats_per_frame = self.num_angles * 3
        self.angle_results_shape = (self.num_frames, self.num_angles, 3)

        # map axes and angles so functions can use results of the current frame
        self.axis_keys = self.default_axis_keys
        self.angle_keys = self.default_angle_keys
        self.axis_mapping = {axis: index for index, axis in enumerate(self.axis_keys)}
        self.angle_mapping = {angle: index for index, angle in enumerate(self.angle_keys)}

        # add non-overridden default pycgm_calc funcs to funcs list
        self.axis_funcs = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAxes().funcs]
        self.angle_funcs = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAngles().funcs]

        # map function names to indices
        self.axis_func_mapping = {function.__name__: index for index, function in enumerate(self.axis_funcs)}
        self.angle_func_mapping = {function.__name__: index for index, function in enumerate(self.angle_funcs)}

        # map function names to the axes they return
        self.axis_result_mapping = {'pelvis_axis': ['Pelvis'],
                                    'hip_axis': ['Hip'], 
                                    'knee_axis': ['RKnee', 'LKnee'],
                                    'ankle_axis': ['RAnkle', 'LAnkle'], 
                                    'foot_axis': ['RFoot', 'LFoot'],
                                    'head_axis': ['Head'],
                                    'thorax_axis': ['Thorax'],
                                    'clav_axis': ['RClav', 'LClav'],
                                    'hum_axis': ['RHum', 'LHum'],
                                    'rad_axis': ['RRad', 'LRad'],
                                    'hand_axis': ['RHand', 'LHand']}

        # map function names to the angles they return
        self.angle_result_mapping = {'pelvis_angle': ['Pelvis'],
                                     'hip_angle': ['RHip', 'LHip'],
                                     'knee_angle': ['RKnee', 'LKnee'],
                                     'ankle_angle': ['RAnkle', 'LAnkle'],
                                     'foot_angle': ['RFoot', 'LFoot'],
                                     'head_angle': ['Head'],
                                     'thorax_angle': ['Thorax'],
                                     'neck_angle': ['Neck'],
                                     'spine_angle': ['Spine'],
                                     'shoulder_angle': ['RShoulder', 'LShoulder'],
                                     'elbow_angle': ['RElbow', 'LElbow'],
                                     'wrist_angle': ['RWrist', 'LWrist']}

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

    def find_marker(self, key):
        # find the slice of a given marker name

        value = None

        try:
            value = self.marker_mapping[key] 
        except KeyError:
            pass

        return value

    def find_measurement(self, key):
        # find the value of a given measurement name

        value = None
        
        try:
            value = self.measurements[key] 
        except KeyError:
            pass

        return value

    def find_axis_index(self, key):
        # find the index of a given axis name

        value = None

        try:
            value = self.axis_mapping[key]
        except KeyError:
            pass

        return value

    def find_angle_index(self, key):
        # find the index of a given angle name

        value = None

        try:
            value = self.angle_mapping[key]
        except KeyError:
            pass

        return value

    def modify_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        # modify an existing function's parameters and returned values
        # used for overriding an existing function's parameters or returned results

        if returns_axes is not None and returns_angles is not None:
            sys.exit('{} must return either an axis or an angle, not both'.format(function))

        # get parameters
        params = []
        for marker in [marker for marker in(markers or [])]:
            params.append(self.find_marker(marker))
        for measurement in [measurement for measurement in(measurements or [])]:
            params.append(self.find_measurement(measurement))
        for axis in [axis for axis in(axes or [])]:
            params.append(self.find_axis_index(axis))
        for angle in [angle for angle in(angles or [])]:
            params.append(self.find_angle_index(angle))
        
        if isinstance(function, str): # make sure a function name is passed
            if function in self.axis_func_mapping:
                self.axis_funcs[self.axis_func_mapping[function]] = getattr(self, function)
                self.axis_func_parameters[self.axis_func_mapping[function]] = params
            elif function in self.angle_func_mapping:
                self.angle_funcs[self.angle_func_mapping[function]] = getattr(self, function)
                self.angle_func_parameters[self.angle_func_mapping[function]] = params
            else:
                sys.exit('Function {} not found'.format(function))
        else:
            sys.exit('Pass the name of the function as a string like so: \'{}\''.format(function.__name__))

        # add returned axes, update
        if returns_axes is not None:
            self.axis_result_mapping[function] = returns_axes
            self.num_axes = len(list(chain(*self.axis_result_mapping.values())))
            self.axis_mapping = {axis: index for index, axis in enumerate(self.axis_keys)}
            self.num_axis_floats_per_frame = self.num_axes * 16
            self.axis_results_shape = (self.num_frames, self.num_axes, 4, 4)

        # add returned angles, update
        if returns_angles is not None:
            self.angle_result_mapping[function] = returns_angles
            self.num_angles = len(list(chain(*self.angle_result_mapping.values())))
            self.angle_mapping = {angle: index for index, angle in enumerate(self.angle_keys)}
            self.num_angle_floats_per_frame = self.num_angles * 3
            self.angle_results_shape = (self.num_frames, self.num_angles, 3)

    def add_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        # add a custom function to pycgm


        # get func object and name
        if isinstance(function,str):
            func_name = function
            func = getattr(self, func_name)
        elif callable(function):
            func_name = function.__name__
            func = function

        if returns_axes is not None and returns_angles is not None:
            sys.exit('{} must return either an axis or an angle, not both'.format(func_name))
        if returns_axes is None and returns_angles is None:
            sys.exit('{} must return a custom axis or angle. if the axis or angle already exists by default, just use self.modify_function()'.format(func_name))

        # get parameters
        params = []
        for marker in [marker for marker in(markers or [])]:
            params.append(self.find_marker(marker))
        for measurement in [measurement for measurement in(measurements or [])]:
            params.append(self.find_measurement(measurement))
        for axis in [axis for axis in(axes or [])]:
            params.append(self.find_axis_index(axis))
        for angle in [angle for angle in(angles or [])]:
            params.append(self.find_angle_index(angle))

        if returns_axes is not None: # extend axes and update
            self.axis_funcs.append(func)
            self.axis_func_mapping = {function.__name__: index for index, function in enumerate(self.axis_funcs)}
            self.axis_func_parameters.append([])
            self.axis_result_mapping[function] = returns_axes
            self.axis_keys.extend(returns_axes)
            self.num_axes = len(list(chain(*self.axis_result_mapping.values())))
            self.axis_mapping = {axis: index for index, axis in enumerate(self.axis_keys)}
            self.num_axis_floats_per_frame = self.num_axes * 16
            self.axis_results_shape = (self.num_frames, self.num_axes, 4, 4)
            
            # set parameters of new function
            self.axis_func_parameters[self.axis_func_mapping[func_name]] = params

        if returns_angles is not None: # extend angles and update 
            self.angle_funcs.append(func)
            self.angle_func_mapping = {function.__name__: index for index, function in enumerate(self.axis_funcs)}
            self.angle_func_parameters.append([])
            self.angle_result_mapping[function] = returns_angles
            self.angle_keys.extend(returns_angles)
            self.num_angles = len(list(chain(*self.angle_result_mapping.values())))
            self.angle_mapping = {axis: index for index, axis in enumerate(self.axis_keys)}
            self.num_angle_floats_per_frame = self.num_angles * 3
            self.angle_results_shape = (self.num_frames, self.num_angles, 3)
            
            # set parameters of new function
            self.angle_func_parameters[self.axis_func_mapping[func_name]] = params

    @property
    def default_angle_keys(self):
        # list of default angle result names

        return ['Pelvis', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle',
                'LAnkle', 'RFoot', 'LFoot',
                'Head', 'Thorax', 'Neck', 'Spine', 'RShoulder', 'LShoulder',
                'RElbow', 'LElbow', 'RWrist', 'LWrist']

    @property
    def default_axis_keys(self):
        # list of default axis result names

        return ['Pelvis', 'Hip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head',
                'Thorax', 'RClav', 'LClav', 'RHum', 'LHum', 'RRad', 'LRad', 'RHand', 'LHand']



    def check_robo_results_accuracy(self, axis_results):
        # test unstructured pelvis axes against existing csv output file
        # this will be removed later

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
        # self.check_robo_results_accuracy(axis_results.reshape(self.axis_results_shape))

        axis_result_keys = list(chain(*self.axis_result_mapping.values()))
        axis_row_dtype = np.dtype([(key, 'f4', (4, 4)) for key in axis_result_keys])

        return np.array([tuple(frame) for frame in axis_results.reshape(self.axis_results_shape)], dtype=axis_row_dtype)

    def structure_trial_angles(self, angle_results):
        # takes a flat array of floats that represent the 3x1 angles at each frame
        # returns a structured array, indexed by result[optional frame slice or index][angle name]

        angle_result_keys = list(chain(*self.angle_result_mapping.values()))
        angle_row_dtype = np.dtype([(key, 'f4', (3,)) for key in angle_result_keys])

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

            for frame in frames:
                flat_axis_results = np.append(flat_axis_results, self.calc(frame)[0].flatten())
                flat_angle_results = np.append(flat_angle_results, self.calc(frame)[1].flatten())

            shared_axes[index_offset * axis_results_size: index_end * axis_results_size] = flat_axis_results
            shared_angles[index_offset * angle_results_size: index_end * angle_results_size] = flat_angle_results

        else:  # single core, just calculate and structure
            for frame in self.marker_data:
                flat_axis_results = np.append(flat_axis_results, self.calc(frame)[0].flatten())
                flat_angle_results = np.append(flat_angle_results, self.calc(frame)[1].flatten())

            # structure flat axis and angle results
            self.axes = self.structure_trial_axes(flat_axis_results)
            self.angles = self.structure_trial_angles(flat_angle_results)

    def calc(self, frame):
        axis_results = []
        angle_results = []

        for index, func in enumerate(self.axis_funcs): # calculate axes
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
