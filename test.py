import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

# Define your class names (must match the training order)
class_names = ["baked potato", "burger", "crispy chicken", "donut", "fries", 
               "hotdog", "pizza", "sandwich", "taco", "taquito"]

def load_and_preprocess(img_path):
    # Load and preprocess image using PIL and EfficientNet's preprocess_input
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)  # Scales to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# Update with the path to your test image
img_path = r"C:\Users\moham\Desktop\4dd1d4df-b833-460c-aa17-4ac95a68ae17_large.jpg"
orig_img, processed_img = load_and_preprocess(img_path)

# 1. Keras Model Prediction
keras_model = tf.keras.models.load_model("food_classifier.h5")
keras_preds = keras_model.predict(processed_img)
keras_class = np.argmax(keras_preds)
print("Keras Model Prediction:", keras_class, "->", class_names[keras_class])

# 2. TFLite Model Prediction
interpreter = tf.lite.Interpreter(model_path="food_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], processed_img)
interpreter.invoke()
tflite_preds = interpreter.get_tensor(output_details[0]['index'])[0]
# Convert raw outputs to softmax probabilities
tflite_probs = np.exp(tflite_preds) / np.sum(np.exp(tflite_preds))
tflite_class = np.argmax(tflite_probs)
print("TFLite Model Prediction:", tflite_class, "->", class_names[tflite_class])

# Display the image with predictions
plt.imshow(orig_img)
plt.title(f"Keras: {class_names[keras_class]} | TFLite: {class_names[tflite_class]}")
plt.axis("off")
plt.show()
