#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
mnistDB=keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnistDB.load_data()
print("shape of x_train",x_train.shape)
print("shape of y_train",y_train.shape)
print("shape of x_test",x_test.shape)
print("shape of y_test",y_test.shape)
import matplotlib.pyplot as plt
plt.imshow(x_train[2],cmap='binary')



# In[4]:


x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
cnn=keras.models.Sequential()
cnn.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=x_train.shape[1:]))
cnn.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
cnn.add(keras.layers.MaxPooling2D(2,2))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(128,activation='relu'))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.Dense(10,activation='relu'))
cnn.summary()


# In[6]:



cnn.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
cnn.fit(x_train,y_train,epochs=1,batch_size=16)
test_loss,test_accuracy=cnn.evaluate(x_test,y_test)


# In[ ]:




