# Multi-Grid detector: data analysis tools & jupyter notebooks

Notebook for analysis of Multi-Grid data. Contains a codebase used by notebooks for the following measurements:

- V20
- Utgård
- LET

## Requisties
- Python3 (https://www.python.org/downloads/)
- Anaconda (https://www.anaconda.com/distribution/)

## Installation
Install dependencies:
```
conda install -c plotly plotly
```

Clone the repository:
```
git clone https://github.com/ess-dg/mg_analysis_notebook.git
```

## Execution
Navigate to mg_analysis_notebook->scripts and enter:
```
jupyter notebook
```
Finally, select the notebook of interest.

## How to create a new project
1. Copy the 'template_notebook.ipynb' script 
2. Rename the project and commit

## Basic usage
1. Execute cell which imports packages
2. Execute cells which define functions
3. Use the 'extract_and_save'-function to extract clusters and events. The function takes a 'run' argument, name of the measurement run, and a 'raw_path' argument, location of the raw data. It then extracts, clusters and save the data in the 'processed folder'.
4. Use the 'load'-function to load clustered data into the notebook, by specifiying a run
5. Define any filters you would like to apply on the data
6. Use the plotting functions to visualize the data

