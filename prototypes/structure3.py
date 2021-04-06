import numpy as np
import sys

sys.path.append("..")
from utils import pycgmIO

class pyCGM():

    def __init__(self, measurements, markers):
        self.measurements = measurements
        self.markers = markers
        self.longest_trial_num_frames = 0

        self.subject_mapping = self.map_subject_data()
        self.measurement_mapping = { measurement_label: index for index, measurement_label in enumerate(self.measurements[0][0])}
        self.marker_mapping = self.map_marker_data()

        print('\n\nWhat the mappings look like:')
        print('\tSubjects:\t', self.subject_mapping)
        print('\tMeasurements:\t', dict(list(self.measurement_mapping.items())[:3]), ' ...')
        print('\tMarkers\t\t', dict(list(self.marker_mapping.items())[:3]), ' ...')

        self.megastructure = self.generate_structure()


    def map_subject_data(self):
        if len(self.measurements[0]) != 2:
            self.measurements = [self.measurements]
            print('Converted measurements to list')

        if not isinstance(self.markers[0], list):
            self.markers = [self.markers]
            print('Converted markers to list')

        self.longest_trial_num_frames = max(len(trial) for trial in self.markers)
        measurement_indices = [i for i in range(len(self.measurements))]
        num_subjects = len(self.measurements)

        subject_mapping = { subject_index: [measurements] for subject_index, measurements in enumerate(measurement_indices) }

        subject_measurement_rows = list(subject_mapping.values())

        trial_indices = [i for i in range(len(self.markers))]
        num_trials = len(self.markers)
        num_trials_per_subject = int(num_trials / num_subjects)
        split_trials = [trial_indices[i:i+num_trials_per_subject] for i in range(0, len(trial_indices), num_trials_per_subject)]
        
        for i in range(len(subject_mapping.keys())):
            subject_mapping[i] += [split_trials[i]]

        
        print('\nSubject mapping logic:')
        print('\tNumber of subjects:', num_subjects)
        print('\tNumber of trials: ', num_trials)
        print('\tNumber of trials per subject: ', num_trials_per_subject, '\n')
        for index, values in subject_mapping.items():
            print('\tSubject %d [measurement index, [trial indices]:' % index, values)
        print('\n')

        return subject_mapping

    def map_marker_data(self):
        markers = self.markers
        xyz_array = np.array([])
        if not isinstance(markers[0], list):
            markers = [self.markers]
            print('Converted markers to list')

        markers = self.markers[0][0]
        marker_keys = list(markers.keys())
        for xyz in markers.values():
            for coord in xyz:
                xyz_array = np.append(xyz_array, coord)

        marker_mapping = {}
        curr_index = 0
        for marker_key in marker_keys:
            marker_mapping[marker_key] = slice(curr_index, curr_index+3, 1)
            curr_index += 3

        print('Marker slice mapping logic:')
        print('\tOriginal markers stored as array([x, y, z])]: ', markers['LFHD'])
        print('\tmarker_mapping[LFHD]: ', marker_mapping['LFHD'])
        print('\n\t# Marker data split up and stored as np.array([x, y, z, x, y, z, ...])')
        print('\txyz_array[marker_mapping[\'LFHD\']]: ', xyz_array[marker_mapping['LFHD']])

        return marker_mapping

    def generate_structure(self):

        def populate_with_subject_measurements():
            print('\nWriting subject measurements to structure...')
            measurement_data_slice = np.zeros_like(megastructure[0][0])
            row_index = 0

            for subject in self.subject_mapping.keys():
                measurement_data_index = self.subject_mapping[subject][0]
                measurement_array = np.array(self.measurements[measurement_data_index][1])

                for index, measurement in enumerate(measurement_array):
                    measurement_data_slice[index] = measurement

                megastructure[0][row_index] = measurement_data_slice
                print('\tWrote subject %d measurements to row %d:' % (subject, row_index))
                row_index += (num_trials_per_subject * 2) + 1

        def populate_with_marker_data():
            print('\nWriting marker data to structure...')
            marker_data_slice = np.zeros_like(megastructure[0][0])
            row_index = 1

            for subject in self.subject_mapping.keys():
                trial_indices = self.subject_mapping[subject][1]

                for trial_index in trial_indices:
                    print('\tProcessing subject %d, trial %d (%d frames) ...' % (subject, trial_index, len(self.markers[trial_index])))
                    for frame_index, frame_data in enumerate(self.markers[trial_index]):
                        frame_as_xyz = np.array([])
                        for array in frame_data.values():
                            for xyz in array:
                                frame_as_xyz = np.append(frame_as_xyz, xyz)
                        offset = megastructure[0][0].shape[0] - frame_as_xyz.shape[0]
                        frame_as_xyz = np.pad(frame_as_xyz, (0, offset))
                        megastructure[frame_index][row_index] = frame_as_xyz
                    print('\t\tWrote marker data to %d layers in row %d' % (frame_index+1, row_index), '\n')
                    row_index += 2 # each trial gets 2 rows (data, results)
                row_index += 1 # skip over subject measurement row in each subject

        num_subjects = len(self.subject_mapping.keys())
        num_trials_per_subject = len(list(self.subject_mapping.values())[0][1])

        num_rows = num_subjects * (1 + num_trials_per_subject*2) # a subject contains: measurements (1) + (num_trials * (trial data (1) + trial results(1))) rows
        num_columns = len(list(self.marker_mapping.values())) * 3 # longest row is the marker data: x,y,z,x,y,z ...
        num_layers = self.longest_trial_num_frames

        megastructure = np.zeros((num_layers, num_rows, num_columns))

        print('\nShape of megastructure:\n[layers] [rows] [columns]')
        print(np.shape(megastructure))

        populate_with_subject_measurements()
        populate_with_marker_data()

        # Checking for equality
        # structure is indexed as [layer] [row] [column]
        print('\nSubject measurements of subject 0:')
        print('\tOriginal: ', self.measurements[0][1][:10], ' ... ')
        print('\tIn struct: ', megastructure[0][0][:10], ' ... ')

        print('\nMarker data of subject 0, trial 0, frame 5432')
        print('\tOriginal: ', list(self.markers[0][5432].values())[:2], ' ... ')
        print('\tIn struct: ', megastructure[5432][1][:6], ' ... ')

        # for the 2 subjects, 3 trials each test
#        print('\nMarker data of subject 1, trial 3, frame 54321')
#        print('\tOriginal: ', list(self.markers[3][54321].values())[:2], ' ... ')
#        print('\tIn struct: ', megastructure[54321][8][:6], ' ... ')

        # for the 2 subjects, 1 trial each test
        print('\nMarker data of subject 1, trial 1, frame 54321')
        print('\tOriginal: ', list(self.markers[1][54321].values())[:2], ' ... ')
        print('\tIn struct: ', megastructure[54321][4][:6], ' ... ')

        return megastructure

measurements_matt = pycgmIO.loadVSK('../SampleData/Sample_2/RoboSM.vsk')
data_matt = pycgmIO.loadData('../SampleData/Sample_2/RoboWalk.c3d')

measurements_other_subject = pycgmIO.loadVSK('../SampleData/59993_Frame/59993_Frame_SM.vsk')
data_other_subject = pycgmIO.loadData('../SampleData/59993_Frame/59993_Frame_Dynamic.c3d')

# 2 subjects, 3 trials each
#print('Testing with 2 subjects, 3 trials per subject')
#measurements = [measurements_matt, measurements_other_subject]
#data = [data_matt, data_matt, data_matt, data_other_subject, data_other_subject, data_other_subject]

# 2 subjects, 1 trial each
print('Testing with 2 subjects, 1 trial per subject')
measurements = [measurements_matt, measurements_other_subject]
data = [data_matt, data_other_subject]

# 1 subject, 2 trials
#print('Testing with 1 subject, 2 trials')
#measurements = measurements_matt
#data = [data_matt, data_other_subject]

# 1 subject, 1 trial
#print('Testing with 1 subject, 1 trial')
#measurements = measurements_matt
#data = data_matt

CGM = pyCGM(measurements, data)

