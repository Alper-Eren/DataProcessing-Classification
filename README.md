# Data Processing and Classification with Python

This repository contains Python code for data processing and classification tasks using various machine learning algorithms. The code focuses on exploring efficiency factors and comparing different estimators to predict the efficiency of a system or process.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data](#data)
- [Code Overview](#code-overview)
- [Results](#results)

## Introduction

The code in this repository aims to process and analyze data to predict the efficiency of a system based on various factors. It includes techniques such as feature selection, principal component analysis (PCA), normalization, and the use of different machine learning algorithms for classification.

## Dependencies

The following dependencies are required to run the code:
- Python 
- Pandas 
- Matplotlib 
- NumPy 
- Scikit-learn 
- SciPy 

## Data

The data used for this project should be provided in a CSV file named 'data.csv'. Ensure that the data follows the following format:

- Each row represents an instance or observation.
- The last column represents the target variable 'Efficiency'.
- The remaining columns represent the features or factors influencing efficiency.

Ensure that categorical variables are converted to numerical representations before running the code.

## Code Overview

The main code is organized as follows:

- Loading the data from a CSV file
- Converting categorical variables to numerical representations
- Splitting the data into features and target variables
- Creating a pipeline for feature selection, PCA, normalization, and classification
- Performing one-hot encoding for categorical variables
- Conducting cross-validation for each estimator
- Evaluating feature selection, PCA transformation, and normalization
- Conducting pairwise t-tests for comparing estimators
- Plotting accuracy scores for each estimator

Refer to the code comments for detailed explanations of each step.

## Results

The code generates visualizations to showcase the results, including boxplots of accuracy scores for different estimators. The results help understand the performance of each estimator and the impact of feature selection, PCA transformation, and normalization on the classification accuracy.

