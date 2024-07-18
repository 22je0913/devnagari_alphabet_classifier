import numpy as np
import keras

test_set = keras.preprocessing.image_dataset_from_directory(
    r".\Test",
    labels="inferred",
    image_size=(32, 32),
    batch_size=138,
)

model = keras.models.load_model(r".\model.keras")

print("\nEdit the paint.png file and press enter in the terminal\n")
input("......")
testPNG = keras.utils.load_img(".\\paint.png")
testPNG = np.resize(testPNG, (32, 32, 3))
testImage = keras.utils.img_to_array(testPNG)
testImage = np.expand_dims(testImage, axis=0)

prediction = np.argmax(model.predict(testImage))
predictionName = test_set.class_names[prediction]

print("Model prediction : ", predictionName)
