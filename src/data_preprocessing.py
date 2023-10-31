import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load images and preprocess them
def load_and_preprocess_images(data_dir, image_size):
    images = []
    labels = []

    for shape in os.listdir(data_dir):  # getting all of the subfolders(shapes)
        shape_dir = os.path.join(data_dir, shape)
        for image_file in os.listdir(shape_dir):    # parsing each image
            image_path = os.path.join(shape_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size)
            image = image.astype(float) / 255.0
            images.append(image)
            labels.append(shape)

    return images, labels

# Split dataset into train, validation, and test sets
def split_dataset(images, labels, test_size=0.15, val_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Encode shape labels into numerical values
def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return labels_encoded