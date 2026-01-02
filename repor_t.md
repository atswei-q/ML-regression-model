# House Price Prediction Project

## Objective
Build a regression model to predict house prices based on property features.

## Dataset
- Housing dataset with 21 features
- Includes numerical and categorical variables

## Methodology
- Data cleaning and preprocessing
- One-hot encoding for categorical features
- Train-test split
- Regression model training

## Model Deployment
- Model saved using joblib
- Feature columns saved to ensure consistency
- FastAPI used to expose prediction endpoint

## Challenges
- Feature mismatch during deployment
- Solved by saving and reusing training feature columns

## Conclusion
This project strengthened my understanding of end-to-end machine learning systems.