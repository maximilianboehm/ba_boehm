Data Augmentation with transformationbased Methods and a Variational Autoencoder
===============================================================================

This project contains the code to augment a simbench-dataset with different methods 
and to evaluate these methods.

### Project structure

    .
    ├── data                   # to store simbench-data and augmented data (is created by the scripts in tools/)
    ├── results                # to store results of evaluation (is created by the evaluation)
    ├── src                    # Source files 
    ├── README.md
    └── requirements.txt
    
### Source Files

This folder contains the following scripts

    .
    ├── tools                       # contains scripts to execute and evaluate single data augmentation methods
        ├── generate_y_data_for_evaluation.py
        ├── run_with_percentage_transformations.py
        ├── run_with_random_shuffle.py
        ├── run_with_vae.py
        └── run_without_da.py
    ├── data_augmentation.py        # functions for da methods
    ├── data_evaluation.py          # function for evaluation
    ├── data_loader.py              # functions for data generation
    ├── timeseries_simulation.py    # functions needed by data_loader.py to generate simbench data
    └── vae.py                      # functions for da with variational autoencoder
    
### Installation
1. Create virtualenv
2. `pip install -r requirements.txt`
3. cd to the directory containing this README and run:
```
export PYTHONPATH=`pwd`:${PYTHONPATH}
```
Step 3 must be repeated every time you start a new shell.

### Usage information

The tools-directory contains the scripts to run data augmentation methods and to evaluate these. The scripts 
can simply be executed from the command-line like:
```
pyhton run_with_vae.py
```
IMPORTANT: tools/generate_y_data_for_evaluation.py needs to be executed before running any of the 
scripts from the tools-directory. This script generates the y_data that is needed for the evaluation.

It is possible to pass optional arguments to the scripts as shown in the following (only examples the values can be changed):

 ```
pyhton run_with_vae.py --times_to_aug 5
python run_with_random_shuffle.py --times_to_aug 5 --splitsize 50
python run_with_percentage_transformations.py --times_to_aug 5 --max_percentage 0.1
```

Times_to_aug sets the size of the resulting dataset (5 means five times larger than the original).
Splitsize sets the size of the sets that the whole dataset is splitted into (50 means 50 rows per set).
Max_percentage sets the maximum percentage that is possible for the percentage transformations 
(0.1 means, that the percentage for the transformations is chosen randomly between 0% and 10% for every value).

The evaluation results of the methods are stored in .txt files in the results-directory.
If the same script is called multiple times, the evaluation results will be appended to the
corresponding files.

### Further usage of the data augmentation methods

The data augmentation methods in data_augmentation.py can be used independently from this project.
Examples for their usage can be seen in the scripts from the tools-directory.
