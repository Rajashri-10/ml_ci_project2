# Model Versioning and Artifact Management

## Objective
To version trained machine learning models and store them as CI artifacts.

## Features
- Each training run saves a versioned model
- Models uploaded as GitHub Actions artifacts
- Performance evaluated automatically

## Tools
- Python
- Scikit-learn
- GitHub Actions

## CI Flow
1. Code push triggers CI
2. Model is trained with version number
3. Model is evaluated
4. Versioned models are stored as artifacts
