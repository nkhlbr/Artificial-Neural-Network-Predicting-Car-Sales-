'''
Make predictions of the Car Purchase value depending on the factors such as Gender, Age, Annual Salary, Credit Card Debt, Net Worth
'''

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

model.summary()

#training weights iwth biases that is why 150 

#press tab to auto complete
model = Sequential()
model.add(Dense(40, input_dim = 5, activation = 'relu' )) #how many neurons in the hidden layer input_dim  = imput columns, 25 neurons 
model.add(Dense(40, activation = 'relu')) #no inputs because the inputs from previous layer are added to this layer
model.add(Dense(1, activation = 'linear')) #for output


#reduce the no.of neurons and observe
#model.add(Dense(5, input_dim = 5, activation = 'relu' )) #how many neurons in the hidden layer input_dim  = imput columns, 25 neurons 
#model.add(Dense(5, activation = 'relu')) #no inputs because the inputs from previous layer are added to this layer
#model.add(Dense(1, activation = 'linear')) #for output

# By reducing the no.of neurons the power has been reduce a higher loss is observed

model.summary()

#increse the number of neurons no. of params increases


# Train the model or fit the model to training data

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#epochs_hist = model.fit(X_train, y_train, epochs = 20, batch_size = 25, verbose = 1, validation_split = 0.2)
#mean square error going down network is learning

#play with epochs

#epochs_hist = model.fit(X_train, y_train, epochs = 20, batch_size = 25, verbose = 1, validation_split = 0.2)

#play with batch size
epochs_hist = model.fit(X_train, y_train, epochs = 20, batch_size = 50, verbose = 1, validation_split = 0.2)



#model evaluation 

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])

#Question what happens when there are variations in no.of epochs, batch size, validation split and no.of neurons



#Now test 
#Gender, Age, Annual Salary, Credit Card Debt, Net Worth

X_test = np.array([[1,50,50000, 10000, 600000]])
y_predict = model.predict(X_test)

print('Expected Purchase Amount', y_predict)
