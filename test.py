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


setup = """
import pycgmIO
from decorator import CGM as decCGM

data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
"""
time = timeit.timeit(
    '[decCGM(frame).pelvis_axis for frame in data]',
    setup=setup,
    number=NUM_RUNS
)
print("=======Decorator approach before modification=======")
print(time)


setup = """
import pycgmIO
from decorator import ModCGM as decModCGM

data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
"""
time = timeit.timeit(
    '[decModCGM(frame).pelvis_axis for frame in data]',
    setup=setup,
    number=NUM_RUNS
)
print("=======Decorator approach after modification=======")
print(time)
