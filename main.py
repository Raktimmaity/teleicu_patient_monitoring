import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import ImageDataGenerator

# Load the dataset
train_dir = 'path/to/train/directory'
validation_dir = 'path/to/validation/directory'
test_dir = 'path/to/test/directory'

# Define the image data generator
datagen = ImageDataGenerator(rescale=1./255)

# Load the training data
train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Load the validation data
validation_generator = datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Define the SVM model
svm = SVC(kernel='linear', C=1)

# Train the model
svm.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model
accuracy = svm.score(validation_generator)
print(f'Validation accuracy: {accuracy:.3f}')

# Save the model
svm.save('doctor_classifier.pkl')