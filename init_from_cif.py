import subprocess
import sys


#this script parse ./path/to/xxx.cif file and generates ./path/to/xxx.conf file

argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python init_conf.py ./path/to/xxx.cif")
    exit(argErrCode)

cif_file_name=str(sys.argv[1])