# Bachelor_code
Code for bachelor
This is the source code for the bachelor project Towards Reinforcement Learning Algorithms for MDPs with Options without Prior Knowledge written by Andreas Eklundh Sørensen
and Asbjørn Risom, supervised by Sadegh Talebi (2023). All of the results in the project can be reproduced running the code in this repo. The code is factored into the two
directories numba_code and non_numba_code. The code in non_numba_code is legacy, and was not used for any results in the thesis, as it is way to slow to run experiments on the
scale we are doing in the project.

The code in numba_code is factored into files 6 main files. The 3 files containing class in the file name is implementation of MDP/SMDP environments as numba classes. The two files UCRL2_L.py and
UCRL_SMDP.py are implementation of the UCRL2-L, SMDP-UCRL-L, SMDP-UCRL and BUS algorithms as numba classes. The files experiment_utils implements a variety of functions, 
both numba and non-numba to easily run and aggregate experiments on the algorithms and environments found in the main files. The code was run on Python >= 3.9.2, and requires numba, numpy, pandas and matplotlib.

Experiments can be replicated by running the files in run_experiment_files via a Python shell. 
