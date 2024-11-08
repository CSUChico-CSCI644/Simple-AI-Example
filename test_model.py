import tensorflow as tf
import csv

# Load the model
model = tf.keras.models.load_model('my_model.keras') 

rows = []
with open('spam_test.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        rows.append(row)

rows = rows[1:] # remove header
labels = [row[0] for row in rows]
texts = [row[1] for row in rows]


# Evaluate the model
loss, accuracy = model.evaluate(texts, labels)

print(f'Loss: {loss}, Accuracy: {accuracy}') 

# Make predictions
predictions = model.predict(labels)
