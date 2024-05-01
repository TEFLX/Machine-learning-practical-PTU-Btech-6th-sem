import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
print("________Ritik kashyap _________")
# Load the dataset
heartDisease = pd.read_csv('heart.csv')
heartDisease.replace('?', np.nan, inplace=True)

print('Few examples from the dataset are given below')
print(heartDisease.head())

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'), ('heartdisease', 'chol')
])

# Learn CPDs using Expectation-Maximization (EM) estimator
print('\nLearning CPD using Expectation-Maximization (EM) estimator')
model.fit(heartDisease, estimator=BayesianEstimator, prior_type='BDeu')

print('Model fitting completed successfully.')

# Perform inference with Bayesian Network
HeartDisease_infer = VariableElimination(model)

# Inferencing examples
print('\nInferencing with Bayesian Network:')
print('\n1. Probability of HeartDisease given Age=28')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q.values)

print('\n2. Probability of HeartDisease given cholesterol=100')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100})
print(q.values)
