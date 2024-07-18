import keras
import matplotlib.pyplot as plt

training_set = keras.preprocessing.image_dataset_from_directory(
    r".\Train",
    labels="inferred",
    image_size=(32, 32),
    batch_size=92,
)

test_set = keras.preprocessing.image_dataset_from_directory(
    r".\Test",
    labels="inferred",
    image_size=(32, 32),
    batch_size=138,
)

classes = training_set.class_names

model = keras.Sequential()
model.add(keras.layers.Input(shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=64, activation="relu", kernel_initializer="uniform"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=46, activation="softmax"))

model.summary()

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

history = model.fit(
    training_set.repeat(),
    steps_per_epoch=85,
    epochs=50,
    validation_data=test_set.repeat(),
    validation_steps=100,
)

plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.2, 1])
plt.legend(loc="lower right")
plt.savefig("graph.jpg")
plt.show()

model.save("model.keras")
