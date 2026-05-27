from datetime import datetime
import sys

import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)
#this script runs diagonalization for plotting band

from plot_energy_band.block_diagonalization import *

argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python run_diagonalization_band_plotting.py /path/to/mc.conf")
    exit(argErrCode)

confFileName = str(sys.argv[1])
num_processes=12
interpolate_point_num=200
verbose=True
out_pickle_file_name=subroutine_eigen_problem_for_energy_band_plot(confFileName,
                                                                   num_processes,
                                                                   interpolate_point_num,verbose)

print(f"out_pickle_file_name={out_pickle_file_name}")