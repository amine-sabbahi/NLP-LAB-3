# NLP Lab 3 - Natural Language Processing with Sklearn

## Overview

This repository contains the code and resources for Lab 3 of the NLP course at Université Abdelmalek Essaadi, Faculté des Sciences et Techniques de Tanger, Département Génie Informatique. The primary objective of this lab is to get familiar with NLP language models using the Sklearn library.

## Objectives

The main purpose behind this lab is to:
1. Establish a preprocessing NLP pipeline.
2. Encode data vectors using various techniques.
3. Train models using different algorithms.
4. Evaluate models using standard metrics.
5. Interpret the obtained results.

## Dataset

1. **Short Answer Grading Dataset**: [answers.csv](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/answers.csv)
2. **Twitter Sentiment Analysis Dataset**: [twitter-entity-sentiment-analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

## Tasks

### Part 1: Language Modeling / Regression

1. Establish a preprocessing NLP pipeline (Tokenization, stemming, lemmatization, stop words removal, etc).
2. Encode your data vectors using Word2Vec (CBOW, Skip Gram), Bag of Words, and TF-IDF.
3. Train your models using SVR, Naive Bayes, Linear Regression, and Decision Tree Algorithms (with Word2Vec embeddings).
4. Evaluate the models using standard metrics (MSE, RMSE, etc), choose the best model and justify your choice.
5. Interpret the obtained results.

### Part 2: Language Modeling / Classification

1. Establish a preprocessing NLP pipeline (Tokenization, stemming, lemmatization, stop words removal, etc).
2. Encode your data vectors using Word2Vec (CBOW, Skip Gram), Bag of Words, and TF-IDF.
3. Train your models using SVM, Naive Bayes, Logistic Regression, and Ada Boosting Algorithms (with Word2Vec embeddings).
4. Evaluate the models using standard metrics (Accuracy, F1 Score, Precision, Recall, etc) and other metrics like BLEU score, choose the best model and justify your choice.
5. Interpret the obtained results.

## Preprocessing Pipeline

The preprocessing steps include:
- Tokenization
- Stemming
- Lemmatization
- Stop Words Removal
- Discretization (if applicable)

## Encoding Techniques

We used the following techniques to encode the text data:
- Bag of Words (BoW)
- TF-IDF
- Word2Vec (CBOW and Skip Gram)

## Models

The following models were trained and evaluated:
- **Regression Models:**
  - SVR (Support Vector Regression)
  - Linear Regression
  - Decision Tree Regressor

- **Classification Models:**
  - SVM (Support Vector Machine)
  - Naive Bayes
  - Logistic Regression
  - AdaBoost

## Evaluation Metrics

We evaluated the models using:
- **Regression Metrics:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

- **Classification Metrics:**
  - Accuracy
  - F1 Score
  - Precision
  - Recall
  - BLEU Score

## Results

### Best Models:

- **Regression (Language Modeling):** 
  - **Linear Regression with CBOW embeddings** - Lowest MSE and RMSE.
  
- **Classification (Sentiment Analysis):**
  - **SVM with Skip Gram embeddings** - Highest Accuracy and F1 Score.

## Conclusion

This lab provided a comprehensive overview of various NLP techniques and word embedding models. We learned how to preprocess text data, represent words as dense vectors, and train and evaluate different machine learning models. These techniques are crucial for numerous NLP applications, including sentiment analysis and document classification.

## Repository Structure

- `data/`: Contains the datasets used in the lab.
- `notebooks/`: Contains Jupyter notebooks with code for each part of the lab.
- `results/`: Contains the evaluation results of the models.
- `README.md`: This file.

## Author

- **SABBAHI Mohamed Amine**

## Acknowledgements

- **Pr. ELAACHAk LOTFI** - Lab Instructor

## Tools

- Google Colab or Kaggle
- GitLab/GitHub
- SpaCy
- NLTK
- Sklearn

## Instructions

1. Clone the repository.
2. Ensure you have the necessary libraries installed:
```
pip install pandas numpy scikit-learn nltk gensim
```
3. Run the notebooks using Jupyterlab or kaggle to reproduce the results.

