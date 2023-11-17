import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('bhsr_model.h5')

# Print the model summary
model.summary()

# Get the output layer of the model
output_layer = model.layers[-1]

# Get the number of classes
num_classes = output_layer.output_shape[-1]

# Assuming 'image_path' is the path to your input image
image_path = '1.jpg'  # Replace with the path to your input image

# Load and preprocess the image
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Make a prediction
predictions = model.predict(img_array)

# Display the predicted probabilities for each class
for i in range(num_classes):
    class_prob = predictions[0][i]
    print(f"Probability for class {i}: {class_prob}")
