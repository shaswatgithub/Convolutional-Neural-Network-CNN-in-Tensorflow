🧠📦 Convolutional Neural Network (CNN) in TensorFlow

Last Updated: 23 Jul, 2025

🖼️ CNNs are the superheroes of computer vision! They learn from images and recognize patterns like edges, shapes, and objects — all without manual programming!

🧰 Libraries Used
Library	Purpose
tensorflow	Deep learning framework 🧠
matplotlib	Plotting training graphs 📈
cifar10	Dataset of 60,000 images across 10 classes 📸
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

🧱 CNN Architecture: Building Blocks
1️⃣ Convolutional Layer

🔍 Learns features like edges, corners, and textures from input images.

layers.Conv2D(32, (3, 3), activation='relu')

2️⃣ Pooling Layer

🔽 Reduces size of feature maps while preserving important features.

Max Pooling – Takes the max value 📈

Average Pooling – Takes the average value 📊

layers.MaxPooling2D(pool_size=(2, 2))

3️⃣ Fully Connected (Dense) Layer

🔗 Connects every neuron from the previous layer to every neuron in the next, helps make the final prediction.

layers.Dense(128, activation='relu')

📦 Dataset: CIFAR-10

A classic dataset for image classification.

🔢 60,000 Images (32x32 RGB)

📂 10 Classes (Airplane, Dog, Cat, etc.)

🧪 Train/Test Split: 50K/10K

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

🏗️ CNN Model Definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

⚙️ Compiling the Model

🧠 Optimizer: Adam

🎯 Loss: categorical_crossentropy for multi-class classification

📊 Metrics: accuracy

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

🏋️ Training the Model

⏳ Epochs: 10

📦 Batch Size: 64

📈 Validation: Done on test set

history = model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=64,
    validation_data=(test_images, test_labels)
)

📊 Evaluating the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")


🧪 Output Example:

Test accuracy: 70.05%
