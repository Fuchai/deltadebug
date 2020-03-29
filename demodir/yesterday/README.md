## How to run
### How to train?
To train the model, your entry point is zero.py

### How to self-play?
The self play experiments can be found in performance.py


## How to run and develop?
### Anaconda 3
Install anaconda 3

### Dependency
Run in command line

```
conda create -n checker
conda activate checker
conda install pytorch numpy=1.14.6
```

### PyCharm setup
Go to settings, Project: PROJECTNAME, Project interpreter, click on the gear icon, select add.

Choose conda environment on the left, select existing environment, select the Python executable for your
checker environment.
To find the Python executable path of your environment, run

```
conda info --envs
```

This command gives you the directory of your environment. In this directory, select ``python.exe``