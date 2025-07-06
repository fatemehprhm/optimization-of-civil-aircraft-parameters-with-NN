import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

column_names = ["Max Pax"	,"Range",	"Take off Weight",	"Empty Weight",	"S gross",	"Thrust",	"Lenght", "Height", "Span"]
df = pd.read_csv('aircraft.csv', names = column_names) 

sp1 = sns.distplot(df['Max Pax'])
sp2 = sns.heatmap(df.corr(),annot = True)
plt.show()

df_norm = (df - df.mean()) / df.std() 

yms = df.to_numpy()
y_mean = yms.mean(axis = 0)
y_mean = np.delete(y_mean, [0,1])
y_std = yms.std(axis = 0)
y_std = np.delete(y_std, [0,1])
def convert_label_value(pred):
    result = np.multiply(pred, y_std) + y_mean
    return result



X = df_norm.iloc[:, :2] 
Y = df_norm.iloc[:,2:9] 

X_arr = X.values
Y_arr = Y.values

print('X_arr shape: ', X_arr.shape)
print('Y_arr shape: ', Y_arr.shape)

X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.15, shuffle = True, random_state=3) 

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)

def get_model():
#kernel_regularizer=regularizers.l2(0.01)
#Dropout(0.2),
    model = Sequential([
        Dense(10, input_shape = (2,), activation = 'tanh'), 
        Dense(25, activation = 'tanh'),  
        Dense(10, activation = 'tanh'),
        Dense(7, activation = 'tanh'),
    ])
    
    model.compile(
        loss='mse',        
        optimizer='SGD'
    )
    
    return model


early_stopping = EarlyStopping(monitor='val_loss', patience = 25) 

model = get_model()

preds_on_untrained = model.predict(X_train)


history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size = 25,
    epochs = 500,
    callbacks = [early_stopping],
    verbose = 0
)

def plot_loss(history):
    h = history.history
    plt.figure(figsize=(8, 8))
    plt.plot(h['val_loss'], label = 'Validation Loss')
    plt.plot(h['loss'], label = 'Training Loss')
    h1 = h['loss'][-1]
    h2 = h['val_loss'][-1]
    print(f'Loss is: {h1:.4f} \nvalidation Loss is: {h2:.4f}\n')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return 

plot_loss(history)

def compare_predictions(preds1, preds2, y_test):
    plt.figure(figsize=(8,8))
    plt.plot(preds1, y_test, 'ro', label='Untrained Model')
    plt.plot(preds2, y_test, 'go', label='Trained Model')
    plt.xlabel('Preds')
    plt.ylabel('Labels')
    
    y_min = min(min(y_test), min(preds1), min(preds2))
    y_max = max(max(y_test), max(preds1), max(preds2))
    
    plt.xlim([y_min, y_max])
    plt.ylim([y_min, y_max])
    plt.plot([y_min, y_max], [y_min, y_max], 'b--')
    plt.legend()
    plt.show()
    return

preds_on_trained = model.predict(X_train)

compare_predictions(preds_on_untrained[:,5], preds_on_trained[:,5], y_train[:,5])


model.save('Opt_model')