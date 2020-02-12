import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from keras.utils import to_categorical
from neural_net.cnn import LeNet
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to dataset directory")
ap.add_argument("-m", "--model", required=True, help="Path to model output directory")
args = vars(ap.parse_args())

if not os.path.exists(args["dataset"]):
    raise ValueError("Path to data directory does not exist!")

X = []
y = []
for (root, dirs, files) in os.walk(args["dataset"]):
    for dir in dirs:
        dir_path = os.path.join(args["dataset"], dir)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)

            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (32, 32))
            X.append(np.expand_dims(np.array(gray), 2))

            label = (file.split(".")[0]).split("_")[1]
            y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

if K.image_data_format() == "channels_first":
    X_train = X_train.reshape((X_train.shape[0], 1, 32, 32))
    X_test = X_test.reshape((X_test.shape[0], 1, 32, 32))
elif K.image_data_format() == "channels_last":
    X_train = X_train.reshape((X_train.shape[0], 32, 32, 1))
    X_test = X_test.reshape((X_test.shape[0], 32, 32, 1))

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = lb.transform(y_test)
y_test = to_categorical(y_test)

opt = SGD(lr=0.01)
model = LeNet.build(width=32, height=32, depth=1, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=1)

predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

model.save(args["model"], overwrite=True)

plt.style.use("ggplot")

figure = plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

figure.savefig("training.png")
