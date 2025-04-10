# Breast Cancer Prediction API

## Overview

This project is a machine learning application that predicts whether a breast tumor is benign or malignant using a GradientBoostingClassifier. The application is built with Flask and includes a modern, user-friendly front-end interface for interacting with the API. Users can input 30 features of a tumor (e.g., mean radius, mean texture) to get a prediction, along with a confidence level and detailed explanation.

The project includes:
- A Flask API for making predictions.
- A front-end interface for user interaction.
- A script to train the model using the Breast Cancer Wisconsin (Diagnostic) dataset.
- A test script to interact with the API programmatically.

## Features

- **Detailed Predictions**: The API provides a prediction (benign or malignant), probability, confidence level, and explanation.
- **Modern Front-End**: A clean, responsive UI with a form to input features and display results.
- **Error Handling**: User-friendly error messages for missing features or prediction failures.
- **Virtual Environment Support**: Runs in an isolated Python environment for dependency management.

## Project Structure
![image](https://github.com/user-attachments/assets/5f9bd013-d220-43f4-983b-463d3e026651)


- `breast_cancer.csv`: The dataset used for training (Breast Cancer Wisconsin dataset).
- `breast_cancer_model.pkl`: The trained model and scaler.
- `breast_cancer_app.py`: The Flask API application.
- `test_breast_cancer_api.py`: Script to test the API programmatically.
- `train_breast_cancer_model.py`: Script to train the model.
- `static/`: Contains CSS and JavaScript files for the front-end.
- `templates/`: Contains the HTML template for the front-end.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Web browser (for the front-end)

## Setup

1. **Clone or Download the Project**:
   Ensure all project files are in a directory (e.g., `C:\Users\muham\Desktop\Lab_task_10\breast_cancer`).

2. **Create a Virtual Environment**:
   Open a terminal and navigate to the project directory:
  ![image](https://github.com/user-attachments/assets/d2e60181-ba99-4cf9-9f0b-0223c3992fc0)

## Running the Project

1. **Train the Model: Run the training script to generate the model:**:
   python train_breast_cancer_model.py
   This will create breast_cancer_model.pkl.
   
3. **Run the Flask API: Start the Flask app:**:
   ![image](https://github.com/user-attachments/assets/fdd2d3e9-b284-44d9-8fd9-961cc9ed6625)
   
3. **Access the Front-End: Open your browser and go to http://localhost:5001. You’ll see the prediction form:**
   1. Enter the 30 features.
   2. Click "Predict" to see the result.
   ![image](https://github.com/user-attachments/assets/e149b836-0adb-4b7c-9e97-75782a8dd74b)

4. **Test the API Programmatically**:
   ![image](https://github.com/user-attachments/assets/f2d19344-4867-4276-b383-9fbf3aaf8c9f)



