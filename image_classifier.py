import tensorflow as tf
import matplotlib.pyplot as plt

print("Loading dataset...")
# Step 2: Dataset Selection
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 3: Data Preprocessing
# Normalize pixel values to the range 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 4: Model Building
# Design a simple Convolutional Neural Network (CNN)
model = tf.keras.models.Sequential([
    # Reshape images (28x28) to fit CNN input requirements
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    
    # Convolution and Pooling layers for feature extraction
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Flatten data and pass to classification layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # 10 output classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Training & Testing
print("\nTraining the model (This may take a minute on CPU)...")
# Epochs=5 means the model will see the dataset 5 times during training
model.fit(x_train, y_train, epochs=5)

print("\nEvaluating model accuracy on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nModel Accuracy: {test_acc*100:.2f}%")

# --- Visualizing the Results ---
print("\nGenerating predictions to show images...")
predictions = model.predict(x_test)

# Create a figure with 5 images
plt.figure(figsize=(12, 4))

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i], cmap='gray') # Display image in grayscale
    
    # Get the number with the highest probability predicted by the model
    predicted_number = predictions[i].argmax() 
    actual_number = y_test[i]
    
    # Show the AI's guess and the actual number above the image
    plt.title(f"AI Guessed: {predicted_number}\nReal: {actual_number}")
    plt.axis('off') # Hide axes for cleaner display

# Display the images in a new window
plt.show()
