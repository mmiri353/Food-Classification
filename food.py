import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from collections import Counter
import random
from PIL import Image

# -------------------------
# 1. Define Paths and Constants
# -------------------------
base_dir = r"C:\Users\moham\Desktop\Food AI\dataset\Fast Food Classification V2"
train_dir = os.path.join(base_dir, "Train")
val_dir   = os.path.join(base_dir, "Valid")
test_dir  = os.path.join(base_dir, "Test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -------------------------
# 2. Load Dataset using image_dataset_from_directory
# -------------------------
train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, 
    labels="inferred", 
    label_mode="int", 
    batch_size=BATCH_SIZE, 
    image_size=IMG_SIZE, 
    shuffle=True, 
    seed=42
)
val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, 
    labels="inferred", 
    label_mode="int", 
    batch_size=BATCH_SIZE, 
    image_size=IMG_SIZE, 
    shuffle=True, 
    seed=42
)
test_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, 
    labels="inferred", 
    label_mode="int", 
    batch_size=BATCH_SIZE, 
    image_size=IMG_SIZE, 
    shuffle=False
)

# Extract class names BEFORE applying any error handling
class_names = train_ds_raw.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds_raw = train_ds_raw.prefetch(buffer_size=AUTOTUNE)
val_ds_raw = val_ds_raw.prefetch(buffer_size=AUTOTUNE)
test_ds_raw = test_ds_raw.prefetch(buffer_size=AUTOTUNE)

# -------------------------
# 3. Custom Data Loader with Safe Image Loading
# -------------------------
def safe_load_and_preprocess_image(file_path, label):
    # Use a Python function (via tf.py_function) to load and preprocess with PIL
    def load_image_fn(path, label):
        try:
            path = path.decode("utf-8")
            img = Image.open(path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img = np.array(img).astype(np.float32)
            # Use EfficientNet's preprocess_input to scale pixel values to [-1, 1]
            img = tf.keras.applications.efficientnet.preprocess_input(img)
        except Exception as e:
            print(f"Skipping invalid image: {path} -- {e}")
            img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            label = -1
        return img, label

    img, lbl = tf.py_function(func=load_image_fn, inp=[file_path, label], Tout=[tf.float32, tf.int32])
    # Set the shape explicitly
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img, lbl

def create_dataset(file_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((list(file_paths.keys()), list(labels.values())))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=42)
    
    dataset = dataset.map(lambda x, y: safe_load_and_preprocess_image(x, y),
                          num_parallel_calls=AUTOTUNE)
    # Filter out images that failed to load (label == -1)
    dataset = dataset.filter(lambda x, y: tf.not_equal(y, -1))
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

# Optionally, if you want to use the custom dataset loader, create label mappings:
def create_label_mapping(directory):
    class_list = sorted(os.listdir(directory))
    label_mapping = {}
    for class_name in class_list:
        class_path = os.path.join(directory, class_name)
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            label_mapping[file_path] = class_list.index(class_name)
    return label_mapping

train_labels = create_label_mapping(train_dir)
val_labels = create_label_mapping(val_dir)
test_labels = create_label_mapping(test_dir)

# Uncomment below to use the custom dataset loader:
# train_ds = create_dataset(train_labels, train_labels, shuffle=True)
# val_ds = create_dataset(val_labels, val_labels, shuffle=True)
# test_ds = create_dataset(test_labels, test_labels, shuffle=False)
# Otherwise, use the raw datasets:
train_ds = train_ds_raw
val_ds = val_ds_raw
test_ds = test_ds_raw

# -------------------------
# 4. Data Augmentation
# -------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

# -------------------------
# 5. Build EfficientNetB0 Model
# -------------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
for layer in base_model.layers[:-10]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# -------------------------
# 6. Compile and Train Model
# -------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

print("Training the model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)

# -------------------------
# 7. Save Model
# -------------------------
model.save("food_classifier.h5")
# For exporting as a TensorFlow SavedModel, use the export API:
model.export("food_classifier_tf")
print("Model saved successfully.")

# -------------------------
# 8. Evaluate Model
# -------------------------
print("Evaluating the model...")
val_loss, val_acc = model.evaluate(val_ds)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# -------------------------
# 9. Example Prediction from Test Directory
# -------------------------
def predict_random_image():
    random_class = random.choice(class_names)
    class_dir = os.path.join(test_dir, random_class)
    random_image = random.choice(os.listdir(class_dir))
    img_path = os.path.join(class_dir, random_image)
    
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
    
    print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

# Predict a random image from the test set
predict_random_image()
