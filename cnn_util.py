from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


def create_classifier():
    classifier = Sequential()

    classifier.add(Conv2D(input_shape=(128, 128, 1), activation='relu', filters=32, kernel_size=(3, 3), padding="same"))
    classifier.add(MaxPool2D(pool_size=(3, 3)))

    classifier.add(Conv2D(activation='relu', filters=64, kernel_size=(3, 3), padding="same"))
    classifier.add(Conv2D(activation='relu', filters=64, kernel_size=(3, 3), padding="same"))
    classifier.add(MaxPool2D(pool_size=(3, 3)))

    classifier.add(Conv2D(activation='relu', filters=128, kernel_size=(3, 3), padding="same"))
    classifier.add(Conv2D(activation='relu', filters=128, kernel_size=(3, 3), padding="same"))
    classifier.add(MaxPool2D(pool_size=(3, 3)))

    classifier.add(Conv2D(activation='relu', filters=256, kernel_size=(3, 3), padding="same"))
    classifier.add(Conv2D(activation='relu', filters=256, kernel_size=(3, 3), padding="same"))

    classifier.add(Conv2D(activation='relu', filters=512, kernel_size=(3, 3), padding="same"))
    classifier.add(Conv2D(activation='relu', filters=512, kernel_size=(3, 3), padding="same"))

    classifier.add(Flatten())
    classifier.add(Dense(units=1568, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=38, activation='softmax'))

    opt = Adam(learning_rate=0.0001)
    classifier.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier
