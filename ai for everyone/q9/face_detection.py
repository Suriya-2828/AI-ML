import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "/Users/suriya/Documents/ai for everyone/q9/dataset/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)
    
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "/Users/suriya/Documents/ai for everyone/q9/dataset/validation",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "/Users/suriya/Documents/ai for everyone/q9/dataset/test",
    target_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False,
    class_mode="binary"
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int)

print(classification_report(
    test_generator.classes,
    predicted_labels,
    target_names=test_generator.class_indices.keys()
))

sample = next(test_generator)

start = time.time()
model.predict(sample[0])
end = time.time()

print("\nInference Time:", end-start, "seconds")

model.save("face_mask_model.keras")
