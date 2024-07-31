# Forest-Fire-Prediction

## Overview

This project aims to predict the occurrence of forest fires using logistic regression. Forest fires can cause significant ecological and economic damage, and early prediction is crucial for timely intervention and prevention. This repository contains the implementation of a logistic regression model to predict the likelihood of forest fires based on various environmental features.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Forest fire prediction involves using machine learning techniques to identify patterns and factors that contribute to the likelihood of a fire occurring. In this project, we use logistic regression, a simple yet effective classification algorithm, to predict forest fires based on historical data. The goal is to provide an early warning system to help prevent and mitigate the impact of forest fires.

## Dataset

The dataset used in this project includes various environmental features such as temperature, humidity, wind speed, and rainfall. These features are crucial in determining the conditions under which forest fires are likely to occur.

### Dataset Features:
- Temperature
- Humidity
- Wind Speed
- Rainfall
- Other relevant environmental factors

The dataset can be found in the `data` directory of this repository.

## Algorithm

### Logistic Regression

Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. In this project, logistic regression is used to model the probability of the occurrence of a forest fire.

**Key Steps in the Algorithm:**
1. **Data Preprocessing:** Cleaning and preparing the dataset for training the model.
2. **Feature Selection:** Identifying the most relevant features that influence forest fire occurrence.
3. **Model Training:** Training the logistic regression model on the prepared dataset.
4. **Model Evaluation:** Evaluating the performance of the model using metrics like accuracy, precision, recall, and F1-score.
5. **Prediction:** Using the trained model to predict the likelihood of forest fires on new data.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/forest-fire-prediction.git
   cd forest-fire-prediction
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset:**
   - Ensure the dataset is in the `data` directory.
   - If using a different dataset, update the data loading section in the script accordingly.

2. **Run the Model:**
   ```bash
   python train_model.py
   ```

3. **Predict Forest Fires:**
   ```bash
   python predict.py
   ```
# **Graph**
![download (1)](https://github.com/user-attachments/assets/8914a3ec-5e60-497c-a6fd-3517e071b91c)

# **HeatMap**

![download (2)](https://github.com/user-attachments/assets/c42c27ef-75ff-4e79-8929-ce4166b5b07e)

# **Pie-Chart**

![download (4)](https://github.com/user-attachments/assets/fff3aa7b-f9c5-4bcc-b730-f1bd7ab44b08)


# **Confusion Matrix**
![download (5)](https://github.com/user-attachments/assets/48036b74-d9ab-4961-8c27-0c0372166d5a)


# **Conslusion**
The Forest Fire Prediction project demonstrates the effective use of logistic regression for predicting the occurrence of forest fires based on environmental features. Through data preprocessing, feature selection, and model training, we have developed a predictive model that can provide early warnings to help mitigate the impact of forest fires.

Feel free to customize this README file as per your specific requirements and project details.
