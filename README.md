# 🍎🍌🍊 Fruit Classification Using Convolutional Neural Networks (CNNs)

## 🌟 Overview

This project focuses on classifying fruits as either fresh or rotten using a Convolutional Neural Network (CNN). The model was trained on a dataset containing images of apples, bananas, and oranges in both fresh and rotten conditions. The goal was to create an efficient model capable of distinguishing between fresh and rotten fruits with high accuracy.

### 🚀 Key Achievements
- **Training Accuracy:** 87.29%
- **Validation Accuracy:** 87.92%
- **Training Loss:** 0.3749
- **Validation Loss:** 0.3672
- **Dataset:** 10,000 images, with a subset of 3,000 images used for training and validation.

## 📂 Project Structure

```
.
├── dataset/
│   ├── train/
│   │   ├── freshapples/
│   │   ├── freshbanana/
│   │   ├── freshoranges/
│   │   ├── rottenapples/
│   │   ├── rottenbanana/
│   │   └── rottenoranges/
│   └── test/
│       ├── freshapples/
│       ├── freshbanana/
│       ├── freshoranges/
│       ├── rottenapples/
│       ├── rottenbanana/
│       └── rottenoranges/
├── fruit_classification.ipynb
├── README.md
└── requirements.txt
```

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fruit-classification-cnn.git
   cd fruit-classification-cnn
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Place the dataset in the `dataset` directory as shown above.

## 🧠 Model Architecture

The CNN model is built using TensorFlow and consists of the following layers:

- **Convolutional Layer 1:** 32 filters, 3x3 kernel size, ReLU activation.
- **MaxPooling Layer 1:** 2x2 pool size.
- **Convolutional Layer 2:** 64 filters, 3x3 kernel size, ReLU activation.
- **MaxPooling Layer 2:** 2x2 pool size.
- **Flatten Layer:** Converts the 2D matrix into a vector.
- **Dense Layer 1:** 128 units, ReLU activation.
- **Output Layer:** 6 units (one for each fruit class), softmax activation.

## 🧩 Data Preprocessing

- **Image Resizing:** All images were resized to 224x224 pixels.
- **Normalization:** Pixel values were normalized to the range [0, 1].
- **One-Hot Encoding:** Labels were one-hot encoded for categorical classification.

## 🏋️‍♂️ Training the Model

The model was trained using the Adam optimizer and categorical cross-entropy loss. The training process ran for 3 epochs with a batch size of 32.

```python
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32)
```

## 📊 Results

After training for 3 epochs, the model achieved:

- **Validation Accuracy:** 87.92%
- **Validation Loss:** 0.3672

### Validation Performance:
![Validation Loss and Accuracy](validation_performance.png)

## 🔍 Testing the Model

You can test the model with new images using the following code snippet:

```python
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
img = image.load_img('/path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict the class of the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Display the image and prediction
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted Category: {predicted_class}")
plt.show()
```

### Example Output:
```bash
Predicted Category: rottenbanana
Predicted probabilities: [[1.1610794e-06 5.7638103e-05 9.3087647e-07 6.5019503e-07 9.9905354e-01 8.8601891e-04]]
```

## 📈 Model Performance

The model effectively distinguishes between fresh and rotten fruits, achieving nearly 88% accuracy. Although trained on a reduced subset, the performance metrics indicate the model’s robustness.

## 🤖 Future Work

- **Data Augmentation:** Implement techniques like rotation, flipping, and zooming to increase dataset size and model robustness.
- **Hyperparameter Tuning:** Experiment with different learning rates, optimizers, and network architectures.
- **Transfer Learning:** Utilize pre-trained models to improve accuracy and reduce training time.

## 📬 Contact

For any questions or feedback, feel free to contact me at [your-email@example.com](mailto:your-email@example.com).
