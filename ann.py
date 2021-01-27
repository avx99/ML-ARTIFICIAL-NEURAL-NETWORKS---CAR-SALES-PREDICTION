import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('Car_Purchasing_Data.csv',encoding='ISO-8859-1')
# sns.pairplot(df)
df = df.drop(['Customer Name','Customer e-mail','Country'],axis = 1)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1,1)
y = scaler.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=1/4)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(25,input_dim = X.shape[1],activation='relu'))
classifier.add(Dense(25,activation='relu'))
classifier.add(Dense(1,activation='linear'))

classifier.summary()

classifier.compile(optimizer = 'adam',loss='mean_squared_error')
hist = classifier.fit(X_train,y_train,epochs=100,batch_size = 10,verbose=1,validation_split=0.2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss during training')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['trainin','validation'])

y_pred = classifier.predict(X_test)
a = 0
for i in range(len(y_pred)):
    a += abs(y_pred[i]-y_test[i])
print("la diff moy est : "+str(a/len(y_pred)))