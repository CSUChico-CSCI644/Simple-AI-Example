import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv

f = open('spam_train.csv', 'r')
rows = []
with open('spam.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        rows.append(row)

rows = rows[1:] # remove header
labels = [row[0] for row in rows]
texts = [row[1] for row in rows]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Create a dummy input layer (we won't use it)
inputs = np.random.rand(len(texts), 10)  # Replace with actual text embeddings if needed

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) # Use 'accuracy' for binary classification


# Train the model (always output 1)
model.fit(inputs, encoded_labels, epochs=1)

# Assuming you have a trained model named 'model'
model.save('my_model.keras')