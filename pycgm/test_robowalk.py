import numpy as np

def check_robo_results_accuracy(axis_results):
    # test structured axes against existing csv output file

    csv_fields = ["PELO", "PELX", "PELY", "PELZ", "HIPO", "HIPX", "HIPY", "HIPZ",
                  "R KNEO", "R KNEX", "R KNEY", "R KNEZ", "L KNEO", "L KNEX", "L KNEY", "L KNEZ",
                  "R ANKO", "R ANKX", "R ANKY", "R ANKZ", "L ANKO", "L ANKX", "L ANKY", "L ANKZ",
                  "R FOOO", "R FOOX", "R FOOY", "R FOOZ", "L FOOO", "L FOOX", "L FOOY", "L FOOZ",
                  "HEAO", "HEAX", "HEAY", "HEAZ", "THOO", "THOX", "THOY", "THOZ", "R CLAO", "R CLAX",
                  "R CLAY", "R CLAZ", "L CLAO", "L CLAX", "L CLAY", "L CLAZ", "R HUMO", "R HUMX",
                  "R HUMY", "R HUMZ", "L HUMO", "L HUMX", "L HUMY", "L HUMZ", "R RADO", "R RADX",
                  "R RADY", "R RADZ", "L RADO", "L RADX", "L RADY", "L RADZ", "R HANO", "R HANX",
                  "R HANY", "R HANZ", "L HANO", "L HANX", "L HANY", "L HANZ"]

    axis_array_fields = ['Pelvis', 'Hip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head',
                         'Thorax', 'RClav', 'LClav', 'RHum', 'LHum', 'RRad', 'LRad', 'RHand', 'LHand']

    # 'Pelvis' in the structured array corresponds to 'PELO', 'PELX', 'PELY', 'PELZ' in the csv (12 values)
    slice_map = { key: slice( index*12, index*12+12, 1) for index, key in enumerate(axis_array_fields) }

    missing_axes = []
    accurate = True

    actual_results = np.genfromtxt( 'SampleData/Sample_2/RoboResults_pycgm.csv', delimiter=',')
    for frame_idx, frame in enumerate(actual_results):
        frame = frame[58:]

        for key, slc in slice_map.items():
            if any(missing_axis in key for missing_axis in missing_axes):
                continue
            else:
                original_o = frame[slc.start    :slc.start + 3]
                original_x = frame[slc.start + 3:slc.start + 6]
                original_y = frame[slc.start + 6:slc.start + 9]
                original_z = frame[slc.start + 9:slc.stop]
                refactored_o = axis_results[frame_idx][key][:3, 3]
                refactored_x = axis_results[frame_idx][key][0, :3] + refactored_o
                refactored_y = axis_results[frame_idx][key][1, :3] + refactored_o
                refactored_z = axis_results[frame_idx][key][2, :3] + refactored_o
                if not np.allclose(original_o, refactored_o):
                    accurate = False
                    error = abs((original_o - refactored_o) / original_o) * 100
                    print(f'{frame_idx}: {key}, origin ({error}%')
                if not np.allclose(original_x, refactored_x):
                    accurate = False
                    error = abs((original_x - refactored_x) / original_x) * 100
                    print(f'{frame_idx}: {key}, x-axis ({error}%')
                if not np.allclose(original_y, refactored_y):
                    accurate = False
                    error = abs((original_y - refactored_y) / original_y) * 100
                    print(f'{frame_idx}: {key}, y-axis ({error}%')
                if not np.allclose(original_z, refactored_z):
                    accurate = False
                    error = abs((original_z - refactored_z) / original_z) * 100
                    print(f'{frame_idx}: {key}, z-axis ({error}%')
    print('All results, not including missing axes, in line with roboresults?', accurate)
