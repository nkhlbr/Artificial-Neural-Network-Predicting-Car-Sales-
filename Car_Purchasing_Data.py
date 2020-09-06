

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')

car_df

#Visualise first five rows
car_df.head(10)

#Visualise last five rows
car_df.tail(5)

#visualize the dataset (creating a 2d matrix)
#Age goes up Car purchase goes up
#Annual salary goes up then Car purchase goes up

sns.pairplot(car_df)

#Data Cleaning

#Drop specific columns, axis = 1 means row drop axis = 0 column drop

X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)

#X =  inputs in the model, y = outputs

#Input
X

#Output
y = car_df['Car Purchase Amount']

y

X.shape #no.of outputs (no.of rows, no.of columns)

y.shape #no.of outputs (no.of rows, no.of columns)

#Data Preprocessing

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) #Normalise the data


X_scaled.shape
X_scaled

scaler.data_max_
scaler.data_min_

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = y.values.reshape(-1,1) #Error is you skip this step
#Error: Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

y_scaled = scaler.fit_transform(y) #Normalise the data



#Train the model


#What happens is test and train have same data
#Test in data have never seen the data in train -- to avoid bias

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.3) #default split between test and train values are 25% if Test_size not included

X_train.shape
X_test.shape




#Artificial Neural Network consists of Input Layer, Atleast One Hidden Layer and Output Layer


#What is Keras
import tensorflow.keras #keras: API that sits on top of tensorflow       
from tensorflow.keras.models import Sequential #built network in a sequential order, just like building Legos
from tensorflow.keras.layers import Dense #All the output coming from a specific layer are fully connected to a next layer in a dense fashion

#press tab to auto complete
model = Sequential()
model.add(Dense(25, input_dim = 5, activation = 'relu' )) #how many neurons in the hidden layer input_dim  = imput columns, 25 neurons 
model.add(Dense(25, activation = 'relu')) #no inputs because the inputs from previous layer are added to this layer
model.add(Dense(1, activation = 'linear')) #for output




