import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset") #downloads dataset from kaggle
print("Path to dataset files:", path)
#kaggle already splits data into Testing/Training for us
TRAIN_DIR = os.path.join(path, "Training")
TEST_DIR  = os.path.join(path, "Testing")

IMG_SIZE = 128   #resizes the images

classes = sorted(os.listdir(TRAIN_DIR))   #creates a 4 element list of each class of tumor (including no tumor)
print("Classes:", classes)

def load_images(directory):
    X =[]
    y = []
    for label, cls in enumerate(classes):
        cls_path = os.path.join(directory, cls) #make directors for each class
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname) #make directories for each individual img
            try:
                img= Image.open(fpath).convert("RGB").resize((IMG_SIZE, IMG_SIZE)) #resizes img to 64x64 and opens
                X.append(np.array(img))
                y.append(label)
            except Exception:
                pass
    return np.array(X), np.array(y) #return np array for each image in dataset with corresponding label

print("\nLoading training images...")
X_train, y_train = load_images(TRAIN_DIR)
print(f"  Total loaded: {len(X_train)}.\n")

print("Loading test images...")
X_test, y_test = load_images(TEST_DIR)
print(f"  Total loaded: {len(X_test)}.\n")


print("X_train shape:", X_train.shape)   # we have 5600 training images, that are each 64x64 with 3 RGB colors
print("X_test shape: ", X_test.shape)    # we have 1600 testing images, that are each 64x64 with 3 RGB colors
print("y_train shape:", y_train.shape)   # we have an array of 5600 classes that correspond to the X_train
print("y_test shape: ", y_test.shape)    # we have an array of 1600 classes that correspond to the y_train

#shapes for each data group



def plot_sample(X, y, index): #used to plot sample images
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

plot_sample(X_train, y_train, 2000) #example
plot_sample(X_train, y_train, 542)

plt.figure(figsize=(15, 15))

for i in range(64): #plot a sample of 64 images to show how they are same dimensions
    plt.subplot(8, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.title(classes[y_train[i]], fontsize=9)

plt.show()

#normalie data/pixels
X_train = X_train / 255.0
X_test  = X_test  / 255.0


#making the ANN model
ann = models.Sequential([
    layers.RandomRotation(0.05), #augmentation
    layers.RandomZoom(0.05), #augmentation
    layers.RandomBrightness(0.05), #vary brightness by 10% for better generalization
    layers.RandomContrast(0.05), #vary contrast by 10% for better generalization
    layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)), #flatten images using 64x64x3
    layers.Dense(3000, activation='relu'), #reduce neurons with ReLU
    layers.Dense(1000, activation='relu'), #reduce again
    layers.Dense(4, activation='softmax')    # produce probalities with confidence of each class using Softmax
])
ann.compile(optimizer='SGD', #using Schotastic Gradient Descent for updating weights
loss='sparse_categorical_crossentropy', #using SCC loss function
metrics=['accuracy']) #displays accuracy on each epoch

ann.fit(X_train, y_train, epochs=5)

y_pred= ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes,
target_names=classes)) #print our classification report to evaluate metrics
ann.save("ann_model.keras") #save our model for backend use



#making the cnn model
cnn = models.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomFlip("horizontal"),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)), #first CONV layer that puts a 3x3 kernel activated by ReLU with 32 filters
    layers.MaxPooling2D((2, 2)), #uses max pooling with a 2x2 filter

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'), #second CONV layer with 64 filters of 3x3 activated by ReLU
    layers.MaxPooling2D((2, 2)), #same max pooling parameter

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(), #flatten data
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4), #dropout layer to prevent overfitting by randomly dropping 50% of neurons
    layers.Dense(4, activation='softmax')    # calculate probabilities of confidence for each class of tumor
])

cnn.compile(optimizer='adam', #using Adam optimizer for updating weights in each epoch
            loss='sparse_categorical_crossentropy', #using SCC loss function
            metrics=['accuracy']) #evaluates on accuracy metrics

cnn.fit(X_train, y_train, epochs=10) #uses 10 forward/backward propogations for increased accuracy

y_predcnn = cnn.predict(X_test)
y_predcnn_classes = [np.argmax(element) for element in y_predcnn]
print("Classification Report: \n", classification_report(y_test, y_predcnn_classes,
target_names=classes)) #print our classification report to evaluate metrics


'''count = 0
#tests each individual point with the actual data, got 1395/1600 correct, 87.5% accuracy
def test_data_point(X_test, y_test, index):
    global count
    if classes[y_predcnn_classes[index]] == classes[y_test[index]]:
      count = count + 1
    plot_sample(X_test, y_test, index)
    print("Predicted:", classes[y_predcnn_classes[index]], classes[y_predcnn_classes[index]] == classes[y_test[index]])
my_range = 1600
for i in range(my_range):
    test_data_point(X_test, y_test, i)
print(f"{count} / {my_range} were marked correctly")'''  #USE THIS CODE TO TEST EACH INDIVIDUAL POINT IN THE TESTING DATASET, IT WILL PLOT THE IMAGE AND PRINT THE PREDICTION AND WHETHER IT WAS CORRECT OR NOT


cnn.save('cnn_model.keras') #save the model for backend use
