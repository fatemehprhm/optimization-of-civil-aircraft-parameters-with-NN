import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

new_model = tf.keras.models.load_model('Opt_model')
new_model.summary()

column_names = ["Max Pax"	,"Range",	"Take off Weight",	"Empty Weight",	"S gross",	"Thrust",	"Lenght", "Height", "Span"]
df = pd.read_csv('aircraft.csv', names = column_names)

yms = df.to_numpy()
y_mean = yms.mean(axis = 0)
y_mean = np.delete(y_mean, [0,1])
y_std = yms.std(axis = 0)
y_std = np.delete(y_std, [0,1])
def convert_label_value(pred):
    result = np.multiply(pred, y_std) + y_mean
    return result

#a = int(input("Please enter pax number: " ))
#b =int(input("Please enter range: " ))
#Input = [a, b]

Input = [179, 4800]
X_test1 = [216.89423, 6015.6923]
X_test2 = np.array(Input) - np.array(X_test1)
X_test3 = np.divide(X_test2,[139.61, 4118.04391])
X_test4 = X_test3.reshape(1,2)

preds_on1 = new_model.predict(X_test4)
preds_on2 = convert_label_value(preds_on1)
preds_on = preds_on2[0]
g = df.iloc[48, 2:].to_numpy()
subtract = preds_on - g
Error = abs(np.divide(subtract, g)*100)

print(f'\nReal values are is:\n\nTake off Weight {g[0]:.2f} tons\nEmpty Weight:{g[1]:.2f} tons \n\
S gross: {g[2]:.2f} m^2\nThrust: {g[3]:.2f} lbs\nLength: {g[4]:.2f} m\nHeight: {g[5]:.2f} m\nSpan: {g[6]:.2f} m')

print(f'\nPredictions are:\n\nTake off Weight {preds_on[0]:.2f} tons\nEmpty Weight:{preds_on[1]:.2f} tons\n\
S gross: {preds_on[2]:.2f} m^2\nThrust: {preds_on[3]:.2f} lbs\nLength: {preds_on[4]:.2f} m\n\
Height: {preds_on[5]:.2f} m\nSpan: {preds_on[6]:.2f} m')

print(f'\nErrors are:\n\nTake off Weight {Error[0]:.2f}%\nEmpty Weight:{Error[1]:.2f}%\n\
S gross: {Error[2]:.2f}%\nThrust: {Error[3]:.2f}%\nLength: {Error[4]:.2f}%\nHeight: {Error[5]:.2f}%\nSpan: {Error[6]:.2f}%')