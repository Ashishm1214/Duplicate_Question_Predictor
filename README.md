# Duplicate Question Prediction

This project aims to predict whether two questions on Quora are duplicates using machine learning techniques. The solution involves data preprocessing, feature engineering, and modeling to achieve reliable predictions. Various models, including Random Forest and XGBoost, are evaluated, and their performance is compared.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Data Description](#data-description)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

## Project Overview

The goal of this project is to develop a machine learning model that can predict whether two given questions are duplicates. To do this, a variety of features are engineered from the text data, including token-based, length-based, and fuzzy matching features. The Random Forest and XGBoost classifiers are evaluated based on their accuracy and confusion matrices.

## Setup and Installation

### Prerequisites

To run this project, you will need:

- Python 3.x
- Libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, `nltk`, `fuzzywuzzy`, `scikit-learn`, `xgboost`, `pickle`

### Files

- **Quora.csv**: The dataset containing pairs of questions and a label indicating whether they are duplicates.
- **modelDQ.pkl**: The saved Random Forest model for predicting duplicate questions.
- **cv.pkl**: The saved CountVectorizer model for text processing.

## Data Description

The dataset includes the following fields:

- `id`: Unique identifier for the question pair.
- `qid1`: ID of the first question.
- `qid2`: ID of the second question.
- `question1`: The first question in the pair.
- `question2`: The second question in the pair.
- `is_duplicate`: A binary label indicating whether the questions are duplicates (1 = duplicate, 0 = not duplicate).

## Feature Engineering

### 1. Basic Features
- Preprocessing: Questions are converted to lowercase, special characters and punctuation are removed, and contractions are expanded.
- Basic Features:
  - Length of the questions in characters and words.
  - Number of common words between the two questions.

### 2. Token-Based Features
- Extract non-stopwords and stopwords from the questions.
- Count common tokens between the two questions.
- Compare the first and last word of the questions for similarity.

### 3. Length-Based Features
- Absolute difference in question lengths.
- Average token length of the questions.
- Ratio of the longest common substring between the two questions.

### 4. Fuzzy Matching Features
- Fuzz ratio, partial ratio, token sort ratio, and token set ratio are computed using the `fuzzywuzzy` library.

### 5. Bag of Words (BOW)
- A CountVectorizer model is used to convert the questions into a bag-of-words representation, limiting to 3000 features.

### 6. Query Point Creation
A function generates a feature vector combining the token-based, length-based, fuzzy, and BOW features for a pair of input questions. This feature vector is then passed to the machine learning model for prediction.

## Modeling

### Random Forest Classifier
- The Random Forest classifier is trained on the engineered features.
- The model achieved an accuracy of 78.92% in predicting duplicate questions.

### XGBoost Classifier
- XGBoost is also used as an alternative to Random Forest for this classification task.
- Both models are compared based on their accuracy and confusion matrix.

### Model Saving
- The trained Random Forest model and CountVectorizer are saved using `pickle` for future use.

## Evaluation

### Random Forest
- **Accuracy**: The Random Forest classifier achieved an accuracy of approximately 78.92%.
- **Confusion Matrix**:
  - True Positives: 1448
  - True Negatives: 3287
  - False Positives: 525
  - False Negatives: 740

### XGBoost
- **Accuracy**: The XGBoost classifier achieved a similar accuracy of approximately 79.47%.
- **Confusion Matrix**:
  - True Positives: 1531
  - True Negatives: 3237
  - False Positives: 575
  - False Negatives: 657

## Usage

To predict whether two questions are duplicates, you can use the pre-trained Random Forest model. The input questions are preprocessed, and their features are extracted using the functions described above. Then, the trained model is used to predict the result, which indicates whether the questions are duplicates.

## Conclusion

This project demonstrates how to apply machine learning techniques to solve the duplicate question detection problem on the Quora dataset. By leveraging a combination of basic, token-based, length-based, and fuzzy matching features, the models achieve solid accuracy. The solution is saved as a pickle file, making it reusable and easily integratable into other systems.
