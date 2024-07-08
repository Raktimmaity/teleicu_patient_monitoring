
# Innovative Monitoring System for TeleICU Patients Using Video Processing and Deep Learning
- Here we use the Machine learning model and Deep learning models.

- Here, we use SVM(Support Vector Machine) for training the model. So, at first we have to know that what is SVM Classifier.

- SVM:In machine learning, support vector machines are supervised max-margin models with associated learning algorithms that analyze data for classification and regression analysis.

- Then we measure accuracy of the model.
- For measure accuracy of the model we have to generate confusion matrix. That hold the some values. These values are -
- TP(True Positive): Correctly predicted positive instances
- TN(True Negative): Correctly predicted negative instances.
- FP(False Positive): Incorrectly predicted Positive instances.
- FN(False Negative): Incorrectly predicted negative instances.
- For measure accuracy of the model the formula is: accuracy = (TP + TN) / (TP + TN + FP + FN)

## Here are the step by step procedure
Here, we perform step by step procedure

### Step 1: Installing libraries
- At first we have to install numpy library 
- Then we have to learn scikit-learn library

### Step 2: Load the dataset
- Then we have to split the dataset
- First we have split the dataset in training dataset
- Then we have to split the dataset into validation 
- Then we have to split the dataset in testing dataset

### Step 3: Define the image generator
- Then we take `ImageDataGenerator` function for image data generator.
- Then we store the data into `datagen` variable.

### Step 4: Load the training data
- Then we load the training data.

### Step 5: Load the validation data
- Then we load the validation data.

### Step 6: Define SVM model
- Here, we use SVM(Support Vector Machine) for tarining the data.
- Sso, here we include SVM Classifier.

### Step 7: Train the model
- Then we train the model using SVM Classifier.

### Step 8: Evaluate the model
- Then we evaluate the model after training the model.
- If there is need some improvement or changes we evaluate the model.
- Here, we generate the actual model accuracy score.

### Step 9: Save the model
- Then we save the model.

### Sample Code
Here is the sample code
``` 
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
```



## Authors

- [Raktim Maity](https://github.com/Raktimmaity)
- [Risav Chatterjee](https://github.com/Raktimmaity)
- [Pooja Maity](https://github.com/Raktimmaity)

