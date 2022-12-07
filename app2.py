#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load our libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Get our training and test data
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')


# In[3]:


#Inspect our training data
train.head()


# In[4]:


#Get our training labels
labels = train['label'].values


# In[5]:


#View the unique labels, 24 in total (no 9)
unique_val = np.array(labels)
np.unique(unique_val)


# In[6]:


#Plot the quantities in each class
plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# In[7]:


#Drop training labels from our training data so we can separate it
train.drop('label', axis = 1, inplace = True)


# In[8]:


#Extract the image data from each row in our csv, remember it's in a row of 748 colums
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])


# In[9]:


#Hot one encode our labels
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)


# In[10]:


#View our labelserr
labels
#View our Labels
#len(labels[0])


# In[11]:


#Inspect an image
index = 2
print(labels[index])
plt.imshow(images[index].reshape(28,28))


# In[12]:


#Use OpenCV to view 10 random images from our training data
import cv2
import numpy as np

for i in range(0,10):
    rand = np.random.randint(0, len(images))
    input_im = images[rand]
    
    sample = input_im.reshape(28,28).astype(np.uint8)
    sample = cv2.resize(sample, None, fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("sample image", sample)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()


# In[13]:


#Split our data into x_train, x_test, y_train and y_test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)


# In[14]:


#Start loading out tensorflow modules and define our batch size etc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

batch_size = 128
num_classes = 24
epochs = 10


# In[15]:


#Scale our images
x_train = x_train / 255
x_test = x_test / 255


# In[16]:


#Reshape them into the size required by TF and Keras
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

plt.imshow(x_train[0].reshape(28,28))


# In[17]:


#Create our CNN Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))


# In[18]:


#Compile our model
model.compile(loss = 'categorical_crossentropy',
             optimizer= Adam(),
             metrics=['accuracy'])


# In[19]:


print(model.summary())


# In[20]:


#Train our model
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


# In[21]:


#Save our model
model.save("sign_mnist_cnn_10_Epochs.h5")
print("Model Saved")


# In[22]:


#View our training history graphically
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()


# In[23]:


#Reshape our test data so that we vcan evaluate it's performance on unseen data
test_labels = test ['label']
test.drop('label', axis = 1, inplace = True)

test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

test_images.shape

y_pred = model.predict(test_images)


# In[24]:


#Get our accuracy score
from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())


# In[25]:


# Create function to match label to letter
# def getletter(result):
#     classLabels = { 0: 'A',
#                     1: 'B',
#                     2: 'C',
#                     3: 'D',
#                     4: 'E',
#                     5: 'F',
#                     6: 'G',
#                     7: 'H',
#                     8: 'I',
#                     9: 'K',
#                     10: 'L',
#                     11: 'M',
#                     12: 'N',
#                     13: 'O',
#                     14: 'P',
#                     15: 'Q',
#                     16: 'R',
#                     17: 'S',
#                     18: 'T',
#                     19: 'U',
#                     20: 'V',
#                     21: 'W',
#                     22: 'X',
#                     23: 'Y'}

def getletter(result):
    classLabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
   
    print (classLabels[result[0]])
    return "result" + classLabels[result[0]]
    # try:
     #   return "result" + classLables[result[0]]
   # except:
     #   return "Error"


# In[26]:


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    ################################
    #frame=cv2.flip(frame, 1)
    
    #define region of interest
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
    
    cv2.imshow('roi sacled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
    
    roi = roi.reshape(1,28,28,1)
    print (np.argmax(np.array(model.predict(roi,1,verbose=0)), axis=1))
    result = str(model.predict(roi, 1, verbose = 0)[0])
    cv2.putText(copy, getletter(np.argmax(np.array(model.predict(roi,1,verbose=0)), axis=1)), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[27]:


model.save("model.h5")


# In[ ]:





# In[ ]:





# In[ ]:




