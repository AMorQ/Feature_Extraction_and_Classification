<!--# Feature_Extraction_and_Classification: Process and classify prostate cancer images via SVGP. Future work will implement SVGPCR-->

# Feature Extraction and Classification

**Description**  
This repository is dedicated to processing and classifying prostate cancer images using Sparse Variational Gaussian Processes (SVGP). Future work will include the implementation of Sparse Variational Gaussian Process Classification Regression (SVGPCR).

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Repository Structure](#repository-structure)
6. [Contributing](#contributing)

## Introduction
Prostate cancer image analysis is a key area in medical imaging research. This repository aims to provide a framework for extracting features and classifying these images using advanced machine learning techniques, specifically utilizing SVGP.

## Features
- **Image Processing**: Tools to preprocess prostate cancer images for feature extraction.
- **Feature Extraction**: Automated feature extraction using deep learning models.
         This part is an implementation of the framework https://github.com/arneschmidt/ssl_and_mil_cancer_classification
        'Efficient Cancer Classification by Coupling Semi Supervised and Multiple Instance Learning'
        This is the implementation of the code of the paper 
        *A. Schmidt, J. Silva-Rodríguez, R. Molina and V. Naranjo, "Efficient Cancer Classification by Coupling Semi Supervised and Multiple Instance Learning," in IEEE      Access, vol. 10, pp. 9763-9773, 2022, doi: 10.1109/ACCESS.2022.3143345.*
- **Classification**: Classify images using Sparse Variational Gaussian Processes (SVGP).
- **Future Work**: Implementation of SVGP Classification for Crowdsourcing (SVGPCR) for enhanced performance, taking advantage to the biggest crowdsourcing dataset for prostate cancer computer vision classification: https://www.sciencedirect.com/science/article/pii/S0169260724004656?via%3Dihub.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/AMorQ/Feature_Extraction_and_Classification.git
2. Navigate to the repository directory:
   ```bash
   cd Feature_Extraction_and_Classification
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
## Usage
(IN ORDER OF UTILIZATION): 
   - main.py: call all functions
   - predata.py: create correct folder structure for keras data generator 
   - data.py: create data generators and call feature extraction models 
   - data_utils.py: helper functions of data.py
   - model_conv.py: feature extraction and classification models 
   - SVGP_utils.py: helper functions of SVGP classification
   - metrics.py: calculate metrics 
   - mlflow_logging: log parameters and metrics to MLFLOW

## Repository Structure
```plaintext
Feature_Extraction_and_Classification/
├── data/                # Directory for storing input data
├── models/              # Contains pre-trained and custom models
├── scripts/             # Scripts for preprocessing and training
├── results/             # Directory for storing output results
├── requirements.txt     # Python dependencies
├── model_conv.py        # Main script for feature extraction and classification
└── README.md            # Repository documentation



