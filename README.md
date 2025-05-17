ML Model Recommendation Pipeline
This project automates the selection of the best machine learning model for tabular datasets using meta-learning. We implement and compare three approaches:

Brute-Force Meta-Labeling: Benchmarks several models per dataset and recommends the best based on meta-features.

Classifier-Based Recommendation: Trains a classifier to predict the best model from dataset meta-features.

Ranker Model Approach: Uses a learning-to-rank model to provide a ranked list of candidate models.


Run the scripts/notebooks:

Brute-Force:
python Project/Brute_Force_Approach_MLModel_Recommendation.py

Classifier:
Open and run Project/Classifier_Approach_MLModel_Recommendation.ipynb

Ranker:
Open and run Project/Ranker_Approach_MLModel_Recommendation.ipynb

Meta-Features Used
Number of instances, features, classes

Feature/instance ratio, class imbalance, entropy

Statistical properties (skewness, kurtosis, correlation, variance)


Authors:
Aditi More
Kusuma Reddyvari
Prajwal Narayanaswamy
Sai Aravind Reddy Katterangandla
CSCI 611 Group 7, Spring 2025
