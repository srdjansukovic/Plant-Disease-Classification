# from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from prepare_images import prepare

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

train_images, train_labels, test_images, test_labels = prepare(root_directory='Dataset')
print(train_images.shape)
print(train_labels.shape)


# train or load model

classifier.fit(train_images, train_labels, batch_size=32, epochs=10)
classifier.save('new_model.h5')
#classifier = load_model('trained_model.h5')


#   if wanted to feed only one image to the network
# test_image = prepare_single_image('digits-smaller/eval/0/2915.jpg')
# images = [test_image]
# prediction = classifier.predict(np.asarray(images))
# print(prediction)

prediction = classifier.predict(test_images)

correct_guesses = 0

with open('output.txt', 'w') as file_handler:
    for item, label in zip(prediction, test_labels):
        winner = list(item).index(max(item))
        file_handler.write("{0} --> {1}\n".format(winner, label))
        if winner == label:
            correct_guesses += 1

overall_percentage = correct_guesses / len(test_labels)
print('Score: ', overall_percentage)
