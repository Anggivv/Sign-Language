#Load our libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route("/index")
def indexku():
    return render_template("index.html")


def gen_frames():  

    #Get our training and test data
    train = pd.read_csv('sign_mnist_train.csv')
    test = pd.read_csv('sign_mnist_test.csv')

    #Get our training labels
    labels = train['label'].values

    #View the unique labels, 24 in total (no 9)
    unique_val = np.array(labels)
    np.unique(unique_val)

    #Plot the quantities in each class
    plt.figure(figsize = (18,8))
    sns.countplot(x =labels)


    #Drop training labels from our training data so we can separate it
    train.drop('label', axis = 1, inplace = True)

    #Extract the image data from each row in our csv, remember it's in a row of 748 colums
    images = train.values
    images = np.array([np.reshape(i, (28, 28)) for i in images])
    images = np.array([i.flatten() for i in images])

    #Hot one encode our labels
    from sklearn.preprocessing import LabelBinarizer

    label_binrizer = LabelBinarizer()
    labels = label_binrizer.fit_transform(labels)




    #Split our data into x_train, x_test, y_train and y_test
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)


    #Start loading out tensorflow modules and define our batch size etc
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

    batch_size = 128
    num_classes = 24
    epochs = 30

    #Scale our images
    x_train = x_train / 255
    x_test = x_test / 255

    #Reshape them into the size required by TF and Keras
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

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


    #Compile our model
    model.compile(loss = 'categorical_crossentropy',
                optimizer= Adam(),
                metrics=['accuracy'])


    #Train our model
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


    #Save our model
    model.save("sign_mnist_cnn_10_Epochs.h5")
    print("Model Saved")

    #View our training history graphically
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'])

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


    #Get our accuracy score
    from sklearn.metrics import accuracy_score

    accuracy_score(test_labels, y_pred.round())


    def getletter(result):
        classLabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    
        print (classLabels[result[0]])
        return "result" + classLabels[result[0]]

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        roi = frame[100:400, 320:620]
        # cv2.imshow('roi', roi)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
            
        # cv2.imshow('roi sacled and gray', roi)
        copy = frame.copy()
        cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
        
        roi = roi.reshape(1,28,28,1)
        print (np.argmax(np.array(model.predict(roi,1,verbose=0)), axis=1))
        result = str(model.predict(roi, 1, verbose = 0)[0])
        cv2.putText(frame, getletter(np.argmax(np.array(model.predict(roi,1,verbose=0)), axis=1)), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        # cv2.imshow('frame', copy)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)