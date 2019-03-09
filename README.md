# Prediction of used cars on ebay

### 1 Introduction 

Simple example code for prediction of used cars on ebay using ML library sklearn and python. Two regressor-algorithms were used: Linear Regressor and Gradient Boosting Regressor. The latter uses an ensemble of Regressors and performs much better.
The data set is included in the project root folder.

### 2 MSE-curve of Gradient Boosting Regressor

The plot below shows how the MSE-loss decreases with increasing number of regressors in the ensemble but stops decreasing at a specific size:

![MSE-curve](https://github.com/saoudh/Machine-Learning-car-value-prediction/blob/master/pred_mse_loss.png)

### 3 Correlation

The following plot shows the positive/negative correlation between some attributes:

![correlation](https://github.com/saoudh/Machine-Learning-car-value-prediction/blob/master/correlation.png)

### 4 Average price per year

The following curve shows average price of the cars each years. Consider the higher value of oldtimer. The reason why the curve falls at year 2019 has to do with sparse data in recent time. There is only one car in the data set from 2019.


![price-per-year-curve](https://github.com/saoudh/Machine-Learning-car-value-prediction/blob/master/prices_per_year.png)
