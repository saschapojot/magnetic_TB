import subprocess
import sys


#this script parse ./path/to/xxx.cif file and generates ./path/to/xxx.conf file

argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python init_conf.py ./path/to/xxx.cif")
    exit(argErrCode)

cif_file_name=str(sys.argv[1])
cif_result=subprocess.run(
    ["python3", "./parse_files/parse_cif.py", cif_file_name],
    capture_output=True,
    text=True
)
# Print the standard output of the execution
print(cif_result.stdout)
# Check if the subprocess ran successfully
if cif_result.returncode != 0:
    print("Error running parse_cif.py:")
    print(cif_result.stderr)
    exit(cif_result.returncode)