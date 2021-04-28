from utils import pycgmIO
from prototypes.decorator_separate_params import gonzCGM, pyCGM  
from numpy import array, random
import c3d

def write_to_file(data, measurements):
    output_file = open('current_data_format.txt', 'w')
    marker_keys = data[0].keys()
    measurement_keys = measurements[0]


    output_file.write('Marker data(per frame):\n\n')
    for key in marker_keys:
        data_format = (key, data[0][key])
        print(data_format)
        output_file.write(str(data_format) + '\n')


    output_file.write('\nMeasurement data:\n\n')
    for index, key in enumerate(measurement_keys):
        data_format = (key, measurements[1][index])
        print(data_format)
        output_file.write(str(data_format) + '\n')

    output_file.close()

def main():
    data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
    measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')

    measurement_keys = measurements[0]
    write_to_file(data, measurements)
    
    # TODO
        # marker slicing by [x, y, z]
#    marker_mapping = { marker: index for index, marker in enumerate(marker_keys) }

#    for key, value in marker_mapping.items():
#        print(key, value)



main()
