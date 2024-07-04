# RainPrediction Project

link :https://will-you-need-an-umbrella.onrender.com/

This project predicts whether it will rain tomorrow using machine learning models. It employs Logistic Regression, XGBoost, and Decision Tree Classifier.

## Files

- `rain_prediction.py`: Contains the `RainPrediction` class.
- `modified_data.csv`: Dataset for training.
- `rain_prediction_model.pkl`: Pickled model.

## Requirements

- pandas
- scikit-learn
- xgboost
- numpy

Install required packages:

```bash
pip install pandas scikit-learn xgboost numpy
```

## Usage

1. **Initialize**: Create an instance of `RainPrediction`.
2. **Train**: Train models using `train_models` method.
3. **Predict**: Use `predict` method with test data.
4. **Save Model**: Save the model using pickle.
5. **Load Model**: Load the saved model using pickle.

## Project Live Link

Check out the live project [here](#).

## Images

### Model Training
![Model Training](images/model_training.png)

### Prediction Output
![Prediction Output](images/prediction_output.png)

## Summary

This project builds and uses multiple classifiers to predict rain, ensuring robust and accurate predictions.
