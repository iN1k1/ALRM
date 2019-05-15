# ALRM
This repos contains the source codes implemented to run the experiments for **person re-identification** used within the paper: "*Accelerated low-rank sparse metric learning for person re-identification*", published in Pattern Recognition Letters, 2018.

# Usage
Before doing any operation, please run the startup file which adds all the directories needed to run the main algorithm
```MATLAB
startup;
```
Once that is done, the MATLAB code runs entirely starting from
```MATLAB
main.m
```
This is the only file that should be executed to partially replicate the experiments.
So far, the current parameters are set to run the experiments on the Market-1501 dataset. If you would like to evaluate a different dataset, please have a look at
```MATLAB
init_paramters.m
```

# Data
The repository does not contain the datasets. To get them in a MATLAB readable format that will directly loaded by the scripts please download them from here:  [Datasets](https://drive.google.com/open?id=1YddJC77is51E9bq8VlI_BQvX2hbeBGNu).

# Thanks
If you use the code contained in this package we appreciate if you'll cite our work. 
>BIBTEX:
@article{Martinel2018a,
author = {Martinel, Niki},
doi = {10.1016/j.patrec.2018.07.033},
issn = {01678655},
journal = {Pattern Recognition Letters},
pages = {234--240},
title = {{Accelerated low-rank sparse metric learning for person re-identification}},
url = {https://doi.org/10.1016/j.patrec.2018.07.033},
volume = {112},
year = {2018}
}


