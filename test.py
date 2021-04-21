import timeit

NUM_RUNS = 5

#setup = """
#from utils import pycgmIO
#from prototypes.cgm import CGM
#
#data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
#"""
#time = timeit.timeit(
#    '[CGM(frame).pelvis_axis for frame in data]',
#    setup=setup,
#    number=NUM_RUNS
#)
#print("=======Original=======")
#print(time)
#
#
#setup = """
#from utils import pycgmIO
#from prototypes.decorator import CGM as decCGM
#
#data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
#"""
#time = timeit.timeit(
#    '[decCGM(frame).pelvis_axis for frame in data]',
#    setup=setup,
#    number=NUM_RUNS
#)
#print("=======Decorator approach before modification=======")
#print(time)
#
#
#setup = """
#from utils import pycgmIO
#from prototypes.decorator import ModCGM as decModCGM
#
#data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
#"""
#time = timeit.timeit(
#    '[decModCGM(frame).pelvis_axis for frame in data]',
#    setup=setup,
#    number=NUM_RUNS
#)
#print("=======Decorator approach after modification=======")
#print(time)
#
#setup = """
#from utils import pycgmIO
#from prototypes.decorator_separate_params import gonzCGM, pyCGM
#
#data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
#measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')
#
#matt = gonzCGM(data, measurements)
#"""
#time1 = timeit.timeit(
#    '''
#matt.run()
#    ''',
#    setup=setup,
#    number=NUM_RUNS
#)
#print('\n===gonzCGM called with decorator and finding/unpacking of custom params===')
#print(time1)
#
#setup = """
#from utils import pycgmIO
#from prototypes.decorator_separate_params import gonzCGM, pyCGM
#from numpy import array, random
#
#data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
#measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')
#
#test1 = random.rand(3)
#test2 = random.rand(3)
#test3 = random.rand(3)
#test4 = random.rand(3)
#test5 = random.rand(3)
#"""
#time2 = timeit.timeit(
#    '''
#for frame in data:
#    pyCGM.calc_pelvis_joint_center(
#        pyCGM,
#        test1,
#        test2,
#        test3,
#        test4,
#        test5)
#    ''',
#    setup=setup,
#    number=NUM_RUNS
#)
#print("=======Calling function directly (with 5 random arrays)======")
#print(time2)
#print('Performance hit of the lookups: {:.2f}'.format((time1-time2)/time2*100), '%')

setup = """
from utils import pycgmIO
from prototypes.pyCGM_structs import pyCGM
from numpy import array, random

data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')

matt = pyCGM(measurements, data)
"""
time = timeit.timeit(
    '''
matt.multi_run(1)
    ''',
    setup=setup,
    number=NUM_RUNS
)

setup = """
from utils import pycgmIO
from prototypes.pyCGM_slices import pyCGM
from numpy import array, random

data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')

matt = pyCGM(measurements, data)
"""
time2 = timeit.timeit(
    '''
matt.multi_run(1)
    ''',
    setup=setup,
    number=NUM_RUNS
)

setup = """
from utils import pycgmIO
from prototypes.pyCGM_structs import pyCGM
from numpy import array, random

data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')

matt = pyCGM(measurements, data)
"""
time3 = timeit.timeit(
    '''
matt.multi_run()
    ''',
    setup=setup,
    number=NUM_RUNS
)

setup = """
from utils import pycgmIO
from prototypes.pyCGM_slices import pyCGM
from numpy import array, random

data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')

matt = pyCGM(measurements, data)
"""
time4 = timeit.timeit(
    '''
matt.multi_run()
    ''',
    setup=setup,
    number=NUM_RUNS
)

print('\nNumber of runs: ', NUM_RUNS)
print("Structuring each frame (1 core): %.2f" % time)
print("Passing known slices (1 core): %.2f" % time2)
print("\nStructuring each frame (multicore): %.2f" % time3)
print("Passing known slices (multicore): %.2f" % time4)

print('\nSeconds difference (1 core): %.2f' % (time - time2))
print('Percentage difference: (1 core): %.2f' % ((time/time2)*100))
print('\nSeconds difference (multicore): %.2f' % (time3 - time4))
print('Percentage difference: (multicore): %.2f' % ((time3/time4)*100))
