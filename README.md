#symmetry analysis for tight-binding, magnetic space group


###########################
Part I, proprocessing
1. python init_from_cif.py ./path/to/xxx.cif, #the xxx.cif file is generated from https://iso.byu.edu/findsym.php
2. #step 1 generates ./path/to/xxx.conf, needs to complete 
    #empty key values in xxx.conf file, then
   python general_script.py ./path/to/xxx.conf
3. #deals with diagonalization of energy bands for plotting
    #should first create BZ_path.conf
    python run_diagonalization_band_plotting.py ./path/to/xxx.conf #same conf file as in step 1,
4. # plotting
    python run_plot_band.py ./path/to/xxx.conf #same conf file as in step 1,