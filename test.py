import timeit

NUM_RUNS = 20

setup = """
import pycgmIO
from cgm import CGM

data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
"""
time = timeit.timeit(
    '[CGM(frame).pelvis_axis for frame in data]',
    setup=setup,
    number=NUM_RUNS
)
print("=======Original=======")
print(time)
