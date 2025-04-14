# Thesis

Creator: John Mario Montoya Zapata

## Version history:
| User                      | Version | date       |
|---------------------------|---------|------------|
| John Mario Montoya Zapata | 0.1.0   | 2025-04-14 |
|                           |         |            |

## Description
Analytical Master's Thesis

## Table of contents
1. [Model for default prediction: Block diagram](#model-for-default-prediction-block-diagram)
2. [Repository structure](#repository-structure)
3. [Cloning this repository](#cloning-this-repository)
4. [Setting up a virtual environment](#setting-up-a-virtual-environment)

## Model for default prediction: Architecture

![diagram](/reports/figures/model_architecture.png)

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
│   └── mlruns/                      <- Folder to store trained, serialized and tracked models using the MLflow library
|
├── notebooks/                       <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                       the creator's initials, and a short `-` delimited description, e.g.
│                                       `101-jmmz-initial-data-exploration.ipynb`.
|
├── references/                      <- Data dictionaries, manuals, and all other explanatory materials
|
├── reports/                         <- Generated analysis as HTML, PDF, LaTeX, etc
│   ├── figures/                     <- Generated graphics and figures to be used in reporting
│   ├── pdfs/                        <- PDF files for reporting
│   └── logs/                        <- Store some flat file reports concerning the execution of commands by terminal mainly
|
├── thesis/             <- Source code for use in this project
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
├── environment.yml                  <- The environment file for reproducing the analysis environment, e.g.
│                                        generated with `conda env export --from-history --file environment.yml`
|
├── requirements.txt                 <- The requirements file for reproducing the analysis environment, e.g.
│                                        generated with `pip freeze > requirements.txt`
|
├── .gitignore                       <- Gitignore file 
|
├── install.md                       <- Instructions to configure virtual environments and install the package as a distributable 
|
├── main.py                          <- Main file to orchestrate re-trains and execution of source code stored in src folder
|
└── run.sh                           <- Executable with predefined commands to run main.py file on a remote server
```

## Cloning this repository.

- To clone this repository using SSH run the next command in your git console
> `git clone path_to_repository/thesis.git`
- To clone this repository using HTTPS run the next command in your git console
> `git clone https://path_to_azure_devops_repositories/thesis.git`

For more details see [Clone a repository](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#clone-a-repository).

## Setting up a virtual environment.

In order to not create conflics between your libraries and the requirements libraries for this project, we highly recomend you to create a new virtual environment to install the requirements libraries in there.

**Check out the installation guide [here](/install.md)**

For more details consult:
- Click [here](https://docs.python.org/3/library/venv.html) to see how to create a virtual environment in python.
- Click [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) if you are using conda.

### Installing and updating project libraries.
The required libraries are listed in the file [`requirements.txt`](/requirements.txt) or [`environment.yml`](/environment.yml). **Please read [the installation guide](/install.md) information for greater detail.**