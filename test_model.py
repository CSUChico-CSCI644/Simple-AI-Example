from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import csv

model = load_model('spam_detector.h5')

data = pd.read_csv('spam_test.csv', encoding='latin-1')
data = data[['v2', 'v1']]
data.columns = ['text', 'label']

# Convert labels to numerical values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=1600)
y = data['label'].values

loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')
