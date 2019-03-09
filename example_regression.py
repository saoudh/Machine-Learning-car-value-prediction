from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import numpy as np

# read data
data = pd.read_csv("Data.csv",encoding='latin-1')
labels=data.price

# select irrelevent attributes including the label attribute "price"
drop_columns = ['price','name','dateCrawled','nrOfPictures','lastSeen','dateCreated','postalCode','offerType','seller','vehicleType','model','monthOfRegistration']
# # select irrelevent attributes
drop_columns_wo_price = ['name','dateCrawled','nrOfPictures','lastSeen','dateCreated','postalCode','offerType','seller','vehicleType','model','monthOfRegistration']

# drop irrelevant columns including the label "price"
data_with_dropped_columns=data.drop(drop_columns, axis=1)
# drop irrelevant columns
data_cleaned_with_price=data.drop(drop_columns_wo_price,axis=1).copy()

# print the data
print(data_with_dropped_columns.head())

#replace date with year
data.yearOfRegistration=pd.to_datetime(data['yearOfRegistration'], format='%Y').dt.year

from sklearn.preprocessing import LabelEncoder,StandardScaler


# transform labels into categorical numbers
lbl_enc = LabelEncoder()
data_with_dropped_columns['gearbox'] = lbl_enc.fit_transform(data_with_dropped_columns['gearbox'].astype(str))
data_with_dropped_columns['fuelType'] = lbl_enc.fit_transform(data_with_dropped_columns['fuelType'].astype(str))
data_with_dropped_columns['brand'] = lbl_enc.fit_transform(data_with_dropped_columns['brand'].astype(str))
data_with_dropped_columns['notRepairedDamage'] = lbl_enc.fit_transform(data_with_dropped_columns['notRepairedDamage'].astype(str))

# group the mean prices by the year
prices_grouped_by_year=data_cleaned_with_price.groupby(['yearOfRegistration', 'powerPS', 'kilometer'], as_index=False).mean().groupby('yearOfRegistration')['price'].mean()

# split cleaned training data into training and test data
x_train , x_test , y_train , y_test = train_test_split(data_with_dropped_columns, labels, test_size = 0.10, random_state =2)

# normalizing the scales of training and test data
normalizer = StandardScaler()
x_train = normalizer.fit_transform(x_train)
x_test = normalizer.transform(x_test)

# training a Linear Regressor
reg=LinearRegression()
reg.fit(x_train,y_train)


# training a gradient Boosting Regressor
gbr_regressor = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 5, min_samples_split = 2,
                                                   learning_rate = 0.1, loss = 'ls')
gbr_regressor.fit(x_train,y_train)

# printing the score of both models
print("Linear Regression - score: ",reg.score(x_train,y_train))
print("Gradient Boosting Regressor - score: ", gbr_regressor.score(x_train, y_train))

losses =[]

y_pred_by_linear_regr = reg.predict(x_test)
for i,y_pred in enumerate(gbr_regressor.staged_predict(x_test)):
    losses.append(gbr_regressor.loss_(y_test, y_pred))

# print error curve of the Gradient Boosting Regressor
plt.figure(figsize=(10, 6))
plt.title("Error rate")
# plot error rate on training data
plt.plot(gbr_regressor.train_score_, 'b-', label='Training Set')
# plot error rate on test data
plt.plot(losses, 'r-', label ='Test Set')

plt.figure(figsize=(10, 6))
# print correlation matrix of the attributes
matrix = data_cleaned_with_price.corr()
plt.title('Correlation')
sns.heatmap(matrix, vmax=0.7, square=True)


# print curve of car price and registration year
plt.figure(figsize=(10, 6))
plt.plot(prices_grouped_by_year)
plt.ylabel('price (€)')
plt.xlabel('Registration year')
plt.title('Car Price / Registration year')
plt.show()
