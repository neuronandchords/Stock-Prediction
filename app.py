import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt


#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
#url="https://github.com/neuronandchords/Exploratary-Music-Analysis-using-Spotipy/blob/master/ACCENTURE.csv"
df = pd.read_csv(r"C:\Users\hp\Downloads\Machine Learning Stuff\Projects\Stock Prediction Full Deployed\Stock Prediction Full Deployed\ACCENTURE.csv")
#print the head
df.head()

df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')
df.index = df['Date']

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Open'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Open'][i] = data['Open'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:300,:]
valid = dataset[300:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
closing_price=closing_price[:101].reshape(1,101)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #int_features = [int(x) for x in request.form.values()]
    #prediction=request.form.values(0)
    output = df

    return render_template('index.html', prediction_text='Opening stocks for selected time span should be {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)

    output = df
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)