
import tensorflow as tf
import numpy as np
import cv2

# Charger le modèle MobileNetV2 pré-entraîné
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Charger et préparer l'image
image_path = "image.jpg"
img = cv2.imread(image_path)
resized = cv2.resize(img, (224, 224))
array = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
array = np.expand_dims(array, axis=0)

# Prédiction
predictions = model.predict(array)
decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

# Afficher les résultats
print("Objets détectés :")
for _, label, confidence in decoded:
    print(f"- {label} : {confidence * 100:.2f}%")
