# thesis installation guide

## Prerequisites

- Python >= 3.11.3

## Create and activate and environment and install required packages

### 1. Using venv library
Go to the project directory and run the following command:

```bash
python -m venv .venv
```

To activate the environment use the following command

On Unix or MacOS:

```bash
source .venv/bin/activate
```

On Windows:

- With gitbash:
```bash
source .venv/Scripts/activate
```
- With command promt or anaconda terminal:
```bash
.venv\Scripts\activate
```

#### 1.1 Install required packages

Please read the requirements.txt file (for virtual environments with venv library) where you will find the suggested libraries, you may need to add or remove libraries.

### Using venv
```bash
python -m pip install --upgrade pip
pip config set global.trusted-host "pypi.org files pythonhosted.org pypi.python.org"
pip install --no-cache-dir -r requirements.txt
```

In case you require Jupyter and Jupyterlab, run the commands:

```bash
pip install jupyter
pip install jupyterlab
```

The packages necessary to run the project are now installed inside the environment.

### Using conda

Please read the environment.yml file (for virtual environments with conda) where you will find the suggested libraries, you may need to add or remove libraries; or you could change the name of the environment or modify the channels to download libraries.

Note that anaconda virtual environments are not stored in the project root folder but are usually stored in the Anaconda/ path. The environment.yml file contains the name of the virtual environment that will be used for this project, however it is possible that you already have a virtual environment with the necessary libraries, so you would not need to create one again but simply use the other one.

#### Create virtual environment and install required packages 

```bash
conda env create --file environment.yml
```

#### Activate virtual environment

```bash
conda activate thesis
```