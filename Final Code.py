## CONCISED WORKING CODE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Load the Data
df = pd.read_csv("Breast_GSE45827.csv")  

# Step 2: Data Preparation
features = df.iloc[:, 2:]  # Gene expression values
target = df['type']  # Cancer types (for classification)

# Step 3: Scale the Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 4: Apply K-means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)  
cluster_labels = kmeans.fit_predict(features_scaled)  

# Step 5: Add Cluster Assignments to Features
features_with_clusters = np.column_stack((features_scaled, cluster_labels)) 

# Step 6: Encode the Target Variable
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Step 7: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    features_with_clusters, target_encoded, test_size=0.2, random_state=42
)

# Step 8: Build the Hybrid Model (CNN + FNN)
input_shape = (features_with_clusters.shape[1], 1)  # Adjusted input shape with clusters

model = keras.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),  # 1D CNN
    layers.MaxPooling1D(2),  # Pooling to reduce dimensionality
    layers.Flatten(),  # Flattening for dense layers
    layers.Dense(256, activation='relu'),  # FNN hidden layer
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for multi-class
])

# Step 9: Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 10: Train the Model
history = model.fit(X_train, y_train, epochs=45, batch_size=50, validation_split=0.2)

# Step 11: Evaluate the Model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  

accuracy = accuracy_score(y_test, y_pred_classes)
conf_matrix = confusion_matrix(y_test, y_pred_classes)
classification_rep = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)

print("Hybrid Model with K-means Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Step 12: Visualize Training Metrics (Loss and Accuracy)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)  
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()  # Adjust subplot spacing
plt.show()

# Step 13: Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Hybrid CNN + FNN with K-means")
plt.show()

# Step 14: Calculate and Visualize Class-Specific Accuracy
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)  
plt.figure(figsize=(10, 6))
sns.barplot(x=label_encoder.classes_, y=class_accuracy * 100)  
plt.xlabel("Cancer Type")
plt.ylabel("Accuracy (%)")
plt.title("Class-Specific Accuracy - With K-means")
plt.show()
