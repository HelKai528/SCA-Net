# SCA-Net: Automatic Estimation of Stellar Atmospheric Physical Parameters Based on Efficient Convolution and Attention Mechanisms

SCA-Net is a tool for automatically estimating stellar atmospheric physical parameters. It is designed based on efficient convolution and attention mechanisms, aiming to provide a convenient and customizable solution for stellar parameter prediction in astronomical research.

Table of Contents

Installation
Usage
Project Structure

Installation

Environment Requirements
To ensure proper program operation, the following environment must be configured:
Python version 3.8
torch version 1.10.0
torchvision version 0.11.1
cv2 version 3.4.10

Installation Steps
Clone the code repository to your local computer.
Navigate to the root directory of the project.
Ensure all files ending with .py have executable permissions by running terminal commands.
Optionally, add the program to the bash environment variables by setting the PYTHONPATH variable and creating a command alias.

Usage

Prediction
The program supports two prediction methods:
The first method is batch prediction by directory: Navigate to the directory containing the files to be processed, then run the prediction command. If the program has been added to the environment variables, you can directly use the predefined command; otherwise, you need to run the Python script directly.
The second method is batch prediction by file list: Prepare a text file listing all the filenames to be processed, then run the prediction command with the file path parameter.

Wavelength Dependency Analysis
You can use the script named window_lxp.py to perform wavelength dependency analysis on the trained model. When running this script, you need to specify the model path and data path parameters.

Project Structure

The project includes the following main files:
scanet.py: This file defines the model code.
insert.py: This file contains the data preprocessing workflow.
label.py: This file contains the label processing workflow.
utils.py: This file contains various utility functions.
window_lxp.py: This file is used for wavelength dependency analysis.

Main Module Description

scanet.py: This file implements the core of the SCA-Net model, designed based on efficient convolution and attention mechanisms.
insert.py: This file is responsible for the data preprocessing workflow, including cleaning, transformation, and formatting of raw data.
label.py: This file is responsible for the label processing workflow, used to prepare label data for training and evaluation.
utils.py: This file contains various auxiliary functions, such as data loaders, evaluation metrics, and visualization tools.
window_lxp.py: This file is used to analyze the performance of the trained model across different wavelength bands.

More features and usage examples will be updated later.
