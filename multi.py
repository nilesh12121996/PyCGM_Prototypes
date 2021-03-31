from prototypes.dec_mapping import CGM
from utils import pycgmIO
import numpy as np
import time
from multiprocessing import Pool, cpu_count, freeze_support


def fun(data, mapping):
    return CGM(data, mapping).pelvis_axis


if __name__ == '__main__':
    freeze_support()
    data = pycgmIO.loadData('SampleData/ROM/Sample_Static.c3d')
    combined = []
    nproc = cpu_count()
    for frame in data:
        mapping = list(frame.keys())
        data = list(frame.values())
        combined.append((data, mapping))

    for n in range(nproc):
        start = time.time()
        with Pool(processes=n + 1) as pool:
            res = pool.starmap(fun, combined)

        print(f"nproc: {nproc}, time: {time.time() - start}")
