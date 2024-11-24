import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

data = pd.read_csv('spam_train.csv', encoding='latin-1')
data = data[['v2', 'v1']]
data.columns = ['text', 'label']

# Convert labels to numerical values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=3900)
y = data['label'].values

# Create a simple model
model = Sequential([
    Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) # Use 'accuracy' for binary classification

X_train = X
y_train = y
# Train the model (always output 1)
model.fit(X_train, y_train, epochs=0)

model.save('spam_detector.h5')