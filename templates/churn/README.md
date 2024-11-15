# Membership Churn Predictor 🎯

A deep learning system that predicts customer churn by analyzing temporal sequences of membership events. Built with Keras using transformer architecture for high-accuracy predictions.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Keras Version](https://img.shields.io/badge/keras-3.0%2B-red)](https://keras.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

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

## Quick Start 🚀

### Prerequisites
- Python 3.10+
- Poetry for dependency management

### Installation
