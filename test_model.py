import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('my_model.keras') 

#TODO load test file in...
# Load your test data
X_test = ...  # Load your test features
y_test = ...  # Load your test labels

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Loss: {loss}, Accuracy: {accuracy}') 

# Make predictions
predictions = model.predict(X_test)
