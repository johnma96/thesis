# 📘 Non-invasive diagnosis of the health status of common beans (Phaseolus vulgaris L) in Colombia: An approach based on spectral fingerprinting and artificial intelligence (Master’s Thesis)

- Author: John Mario Montoya Zapata
- Degree: Master’s Thesis
- Field: Precision Agriculture · Hyperspectral Imaging · Machine Learning & Deep Learning

## Version history:
| User                      | Version | date       |
|---------------------------|---------|------------|
| John Mario Montoya Zapata | 0.1.0   | 2025-04-14 |
| John Mario Montoya Zapata | 1.0.0   | 2026-01-22 |
|                           |         |            |

## 📌 Overview

This repository contains the complete codebase, data management structure, experiments, and documentation developed for the master’s thesis focused on detecting phosphorus (P) deficiency stress in common bean (Phaseolus vulgaris L.) using UAV-based hyperspectral imagery and machine learning / deep learning techniques.

The project addresses a binary classification problem under real field conditions, combining:

- Hyperspectral data preprocessing and band selection
- Spectral and spectral–spatial modeling
- Robust experimental design with spatial data splitting
- Model tracking, optimization, and reproducibility

## 🧪 Scientific Contributions

- End-to-end workflow for handling large hyperspectral datasets using Zarr + DVC
- Informed spectral band selection using SNR proxy and spectral decorrelation analysis
- Comparison of ML vs DL approaches, including CNN-1D and CNN-2D
- Demonstration of the importance of spatial context in nutrient stress detection
- Reproducible experimentation with MLflow, Optuna, and DVC

## Table of contents
1. [Model for stress prediction: Architecture CNN-2D](#model-for-stress-prediction-architecture-cnn-2d)
2. [Repository structure](#repository-structure)
3. [🔬 Data Management & Reproducibility](#data-management-and-reproducibility)
4. [📚 Annexes and Technical Reports](#annexes-and-technical-reports)
5. [🧠 Reproducibility Statement](#reproducibility-statement)
6. [Cloning this repository](#cloning-this-repository)
7. [Setting up a virtual environment](#setting-up-a-virtual-environment)
8. [📄 License](#license)
9. [📬 Contact](#contact)

## Model for stress prediction: Architecture CNN-2D

![diagram](/reports/figures/arquitectura_cnn2d.png)


## Repository structure.

This project structure was partially influenced by the [Cookiecutter Data Science project](https://drivendata.github.io/cookiecutter-data-science/) and [reproducible-model](https://github.com/cmawer/reproducible-model) repository.

Check this [post](https://www.jeremyjordan.me/ml-projects-guide/) by Jeremy Jordan for get guidelines on managing ML projects.

Other resources.
- Books
    - [Clean Machine Learning Code](https://leanpub.com/cleanmachinelearningcode)

```
├── LICENSE
|
├── README.md                        <- You are here
|
├── .dvc/                            <- Folder with remote source configuration for the DVC library
|
├── app/                             <- Folder to store the API that exposes the model
|
├── credentials/                     <- Folder to store credentials files
|
├── data/                            <- Folder that contains data used or generated
│   ├── external/                    <- Data from third parties (external to the core company of the project)
│   ├── interim/                     <- Data in an intermediate state of processing
│   ├── processed/                   <- Data fully processed and ready to be used in modeling
│   └── raw/                         <- The original, immutable data dump
|
├── docs/                            <- A default Sphinx project; see sphinx-doc.org for details
|
├── models/                          <- Trained and serialized models, model predictions, or model summaries
|
├── notebooks/                       <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                       the creator's initials, and a short `-` delimited description, e.g.
│                                       `101-jmmz-initial-data-exploration.ipynb`.
|
├── references/                      <- Data dictionaries, manuals, and all other explanatory materials
│   ├── others/                      <- Generated graphics and figures to be used in reporting
│   ├── papers/                      <- Scientific papers used as bibliographic sources
│   └── logs/                        <- Store some flat file reports concerning the execution of commands by terminal mainly
|
├── reports/                         <- Generated analysis as HTML, PDF, LaTeX, etc
│   ├── figures/                     <- Generated graphics and figures to be used in reporting
│   ├── pdfs/                        <- PDF files for reporting
│   └── logs/                        <- Store some flat file reports concerning the execution of commands by terminal mainly
|
├── spectralcrop/                    <- Source code for use in this project
│   ├── __init__.py                  <- Makes src a Python module
│   ├── archive/                     <- Old scripts that are removed by restructuring the code. They are kept for future reference
│   ├── data/                        <- Scripts to generate, obtain, clean or load raw data
│   ├── features/                    <- Scripts to turn data into features for modeling
│   ├── models/                      <- Scripts to use trained models to make predictions and to retrain models
│   ├── performance/                 <- Scripts to evaluate the performance of models and to calculate metrics from the trained model 
│   ├── visualization/               <- Scripts to generate evaluation graphs or reports 
│   └── utils/
│      ├── __init__.py
│      ├── absolute_paths.py         <- Module for handling absolute path
│      ├── make_connection.py        <- Module to generate connections to different RDBMS
│      ├── read_sql_file.py          <- Module for queries stored in .sql files
│      ├── repair_str_columns.py     <- Module to repair columns transformed from string to numbers
│      ├── save_data.py              <- Module to ingest data to different RDBMS
│      ├── utilities.py              <- Module with utility functions of the package
│      └── load_data.py              <- Module for reading data from different RDBMS
│
├── queries/                         <- Folder to store .sql files used at some point in the modeling process   
│   ├── develop/                     <- Queries created in bulding process 
│   └── production/                  <- Clean queries used to production
|
├── tests/                           <- Unit and integration tests
|
├── environment.yml                  <- The environment file for reproducing the analysis environment, e.g.
│                                        generated with `conda env export --from-history --file environment.yml`
|
├── requirements*.txt                 <- The requirements file for reproducing the analysis environment, e.g.
│                                        generated with `pip freeze > requirements.txt`
|
├── .gitignore                       <- Gitignore file 
|
├── main.py                          <- Main file to orchestrate re-trains and execution of source code stored in src folder
|
└── run.sh                           <- Executable with predefined commands to run main.py file on a remote server
```

## Data Management and Reproducibility

- DVC is used for versioning large datasets and trained models
- MLflow tracks experiments, metrics, and artifacts
- Optuna enables Bayesian hyperparameter optimization
- Zarr allows scalable storage of hyperspectral data cubes

**Remote DVC repository:** https://dagshub.com/johnma96/thesis.s3

## Annexes and Technical Reports

- Annex 32: Spectral data acquisition and field experiment
- Annex 41: Multispectral and hyperspectral image processing

Available at: references/technical_reports/

### 🚀 Configuración de DVC con Google Drive (OAuth personalizado)
Scientific papers used You can refer to the description in the [official source](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive) and the information mentioned in this [stackoverflow] thread (https://stackoverflow.com/questions/75454425/access-blocked-project-has-not-completed-the-google-verification-process) (It is important to look at the comments as they describe the latest GCP graphical interface). You can also consult the step-by-step information [here](docs/README_DVC_GoogleDrive.md) as a bibliographic source.


## Reproducibility Statement

All results reported in the thesis can be reproduced using:

- Versioned datasets and models (DVC)
- Fixed random seeds
- Logged hyperparameters and metrics (MLflow)
- Explicit environment definitions


## Cloning this repository.

- To clone this repository using SSH run the next command in your git console
> `git clone git@github.com:johnma96/thesis.git`
- To clone this repository using HTTPS run the next command in your git console
> `git clone https://github.com/johnma96`

For more details see [Clone a repository](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#clone-a-repository).

This repository is linked to the [DagsHub](https://dagshub.com/johnma96/thesis) (https://dagshub.com/johnma96/thesis) platform, where data is stored using DVC and models are tracked with MLflow.

## Setting up a virtual environment.

In order to not create conflics between your libraries and the requirements libraries for this project, we highly recomend you to create a new virtual environment to install the requirements libraries in there.

**Check out the installation guide [here](/install.md)**

For more details consult:
- Click [here](https://docs.python.org/3/library/venv.html) to see how to create a virtual environment in python.
- Click [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) if you are using conda.

### Installing and updating project libraries.
The required libraries are listed in the file [`requirements.txt`](/requirements.txt), [`requirements-pytorch-cpu.txt`](/requirements-pytorch-cpu.txt) and [`requirements-pytorch-cu126.txt`](/requirements-pytorch-cu126.txt). **Please read [the installation guide](/install.md) information for greater detail.**

## License

### Code
The source code in this repository is licensed under the [MIT License](/LICENSE).

### Data, Labels and Experimental Results
All datasets, labels, trained models, figures, and experimental results are **NOT open**.
They are protected under an ["All Rights Reserved" license](/DATA_LICENSE.md) and may not be reused
without explicit permission from the author.

## Contact

- John Mario Montoya Zapata
- Data scientist, MLOps engineer, Master's degree in analytical engineering 
- 📧 jmmontoyaz@unal.edu.co, jmmontoyaz13@gmail.com
- 🔗 [GitHub](https://github.com/johnma96) / [LinkedIn](https://www.linkedin.com/in/john-m-montoya-z-1a375a15b/)