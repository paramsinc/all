# Membership Churn Predictor 🎯

A deep learning system that predicts customer churn by analyzing temporal sequences of membership events. Built with Keras using transformer architecture for high-accuracy predictions.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Keras Version](https://img.shields.io/badge/keras-3.6%2B-red)](https://keras.io/)

## Overview

This project helps businesses predict which customers are likely to cancel their memberships by analyzing sequences of customer interactions and membership events. It uses a transformer-based architecture to detect complex patterns in temporal event data.

### Key Features
- Processes temporal sequences of membership events (visits, renewals, etc.)
- Handles variable-length event sequences through padding
- Combines event type embeddings with time-difference features
- Uses transformer architecture for pattern detection
- Provides real-time predictions through FastAPI
- Includes comprehensive model evaluation tools

## Technical Architecture

### Data Processing (`data.py`)
- Supports multiple event types: new memberships, renewals, washes, etc.
- Processes raw CSV data from multiple sources (transactions, memberships, churn events)
- Handles temporal sequence creation and normalization
- Implements padding and scaling for variable-length sequences

### Model Architecture (`model.py`)
- Event type embedding layer
- Time difference feature integration
- Transformer encoder layers with configurable parameters
- Attention-based sequence processing
- Binary classification output for churn prediction

### Training Pipeline (`train.py`)
- Configurable train/test split
- Batch processing with progress tracking
- Validation data monitoring
- Model checkpoint saving

### Evaluation Tools (`eval.py`)
- Classification metrics (precision, recall, F1)
- ROC and PR curves
- Confusion matrix visualization
- Feature importance analysis
- Learning curve monitoring

### API Service (`serve.py`)
- FastAPI-based REST endpoint
- Real-time prediction serving
- Request timing middleware
- Batch prediction support

### Configuration (`config.py`)
- Configurable hyperparameters for training and evaluation
- Paths for data and model storage

## Quick Start 🚀

### Prerequisites
- Python 3.10+
- Poetry for dependency management

### Installation
1. `git clone https://github.com/paramsinc/all.git`
2. `cd all/templates/churn`
3. `poetry shell`
4. `poetry install`

### Usage
- Add the path to your data files and other variables to `config.py`
- The goal is to create an ordered sequence of events for each membership. For this example, there are three files that are read in and formatted (`raw_transactions.csv`, `raw_churn_events.csv`, `raw_memberships.csv`). From these files, we can build the sequence of events for each membership. The format of `raw_transactions.csv` is:
  ```
  MEMBERSHIP_ID,CREATED_AT
  001,4/30/2023  7:00 PM
  002,5/02/2023  4:00 PM
  ```
  The format of `raw_churn_events.csv` is:
    ```
    MEMBERSHIP_ID,CHURN_DATE,CHURN_TYPE
    001,6/10/2024,terminated
    002,6/12/2024,expired
    ```
  The format of `raw_memberships.csv` is:
    ```
    MEMBERSHIP_ID,CREATED_AT,TRANSACTION_CATEGORY
    001,4/30/2023  7:00 PM,renewed membership
    002,5/02/2023  4:00 PM,wash
    ```
- Run `python main.py` to train the model and save it to the `prod_model_path` from `config.py`
- Run `python serve.py`, which reads in the model you just trained, to start serving real-time churn predictions
  - `serve.py` takes in a `sequences` argument, which is a list of lists of events with their timestamps.