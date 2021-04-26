from utils import pycgmIO
from custom_CGMs import harrington_hip_CGM, oxfordCGM
from pyCGM import pyCGM
from SM import SubjectManager
import sys

measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')
marker_data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
static_trial = pycgmIO.loadData('SampleData/Sample_2/RoboStatic.c3d')
oxford_marker_data = pycgmIO.loadData('SampleData/Oxford/OFM_Walk1.c3d')

def default_cgm_example():
    # running one subject, giving each core a block of frames to calculate

    subject = pyCGM(measurements, static_trial, marker_data)
    subject.multi_run()
    subject_pelvis = subject.axes['Pelvis'] # subject's pelvis axis at each frame
    subject_pelvis_axis_frame_400 = subject.axes[400]['Pelvis'] # subject's pelvis axis at frame 400
    subject_pelvis_angle_frame_400 = subject.angles[400]['Pelvis'] # subject's pelvis angle at frame 400

def harrington_cgm_example():
    # running one subject with an overridden hip axis calculation

    harrington_subject = harrington_hip_CGM(measurements, static_trial, marker_data)
    harrington_subject.multi_run()
    harrington_pelvis_frame_400 = harrington_subject.axes[400]['Hip']

def oxford_foot_example():
    # running one subject with a custom function that returns custom axes

    oxford_subject = oxfordCGM(measurements, static_trial, oxford_marker_data)
    oxford_subject.add_function('oxford_foot', axes=['forefoot_left', 'forefoot_right', 'hindfoot_left', 'hindfoot_right'])
    oxford_subject.run()
    subject_forefoot_left = oxford_subject.axes['forefoot_left']
    subject_hindfoot_right = oxford_subject.axes['hindfoot_right']

def multiple_similar_subjects_example():
    # run 9 default trials in parallel, retrieve specific results

    subject_1 = pyCGM(measurements, static_trial, marker_data)
    subject_2 = pyCGM(measurements, static_trial, marker_data)
    subject_3 = pyCGM(measurements, static_trial, marker_data)

    subjects = SubjectManager([subject_1, subject_2, subject_3, 
                               subject_1, subject_2, subject_3, 
                               subject_1, subject_2, subject_3])
    subjects.multi_run()

    all_pelvis = subjects.joint_axis('Pelvis') # pelvis axes of all subjects, all frames
    pelvis_frames_5_15 = subjects.joint_axis('Pelvis', 5, 15) # pelvis axes of all subjects, frames 5-15
    left_ankle_frames_5_15 = subjects.joint_angle('L Ankle', 5, 15) # left ankle angle of all subjects, frames 5-15

def multiple_custom_subjects_example():
    # run multiple custom trials in parallel, retrieve specific results

    default_subject = pyCGM(measurements, static_trial, marker_data)

    harrington_subject = harrington_hip_CGM(measurements, static_trial, marker_data)

    oxford_subject = oxfordCGM(measurements, static_trial, oxford_marker_data)
    oxford_subject.add_function('oxford_foot', axes=['forefoot_left', 'forefoot_right', 'hindfoot_left', 'hindfoot_right'])

    various_subjects = SubjectManager([default_subject, harrington_subject, oxford_subject])
    various_subjects.multi_run()

    all_pelvis = various_subjects.joint_axis('Pelvis') # pelvis axes of all subjects, all frames
    pelvis_frames_5_15 = various_subjects.joint_axis('Pelvis', 5, 15) # pelvis axes of all subjects, frames 5-15
    left_ankle_frames_5_15 = various_subjects.joint_angle('L Ankle', 5, 15) # left ankle angle of all subjects, frames 5-15
    oxford_hindfoot_left_100 = oxford_subject.axes[100]['hindfoot_left'] # left hindfoot axis of oxford trial

default_cgm_example()
harrington_cgm_example()
oxford_foot_example()
multiple_similar_subjects_example()
multiple_custom_subjects_example()