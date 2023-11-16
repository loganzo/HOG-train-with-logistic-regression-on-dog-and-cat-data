import os
import random
import cv2 as cv
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gradio as gr
import sqlite3

# Function to create SQLite database and table
def create_database(db_path="predictions.db"):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, image_name TEXT, prediction TEXT)")
    connection.commit()
    connection.close()

# Function to save prediction to SQLite database
def save_to_database(image_name, prediction, db_path="predictions.db"):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("INSERT INTO predictions (image_name, prediction) VALUES (?, ?)", (image_name, prediction))
    connection.commit()
    connection.close()

def random_image_index(length_data: int) -> int:
    return np.random.randint(0, length_data - 1)

def random_images(data_path: str) -> list:
    images_lst = []
    images_root = os.listdir(data_path)

    for i in range(16):
        index_image = random_image_index(len(images_root))
        image_path = os.path.join(data_path, images_root[index_image])
        images_lst.append(image_path)

    return images_lst

def image_flatten(image: np.array, image_size: int, augmentation: bool = False) -> np.array:
    image = cv.resize(image, (image_size, image_size))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalized_image = cv.equalizeHist(image)

    if augmentation:
        if random.random() > 0.5:
            equalized_image = cv.flip(equalized_image, 1)

    features = hog(equalized_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

def create_dataset(data_path: str, image_size: int) -> list:
    data_set = []

    for label in os.listdir(data_path):
        label_folder = os.path.join(data_path, label)

        for image_name in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_name)
            image = cv.imread(image_path)

            if image is not None:
                features = image_flatten(image, image_size)
                if label == "dogs":
                    data_set.append((features, image_name, 1))
                else:
                    data_set.append((features, image_name, 0))
            else:
                print(f"Can't open image {image_path}")

    random.shuffle(data_set)
    X, image_names, y = zip(*data_set)
    X = np.array(X)
    image_names = np.array(image_names)
    y = np.array(y)

    return X, image_names, y

# Gradio interface function
def predict_image(img, image_name):
    # Convert Gradio Image object to NumPy array
    img_array = img.astype(np.uint8)

    # Convert to BGR (OpenCV) format
    img_bgr = cv.cvtColor(img_array, cv.COLOR_RGBA2BGR)

    # Resize and preprocess image
    img_features = image_flatten(img_bgr, 128)

    # Make prediction using the trained model
    prediction = model.predict([img_features])[0]
    prediction_label = "Dog" if prediction == 1 else "Cat"

    # Save prediction to SQLite database
    save_to_database(image_name, prediction_label)

    return prediction_label

# Main function
if __name__ == "__main__":
    # Create SQLite database and table
    create_database()

    # Create dataset
    X_train, image_names_train, y_train = create_dataset("D:/try hard/Source/Source code/small_dog_cat_dataset-master/small_dog_cat_dataset-master/train", 128)
    X_test, image_names_test, y_test = create_dataset("D:/try hard/Source/Source code/small_dog_cat_dataset-master/small_dog_cat_dataset-master/test", 128)

    # Model train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Model test
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Gradio interface
    iface = gr.Interface(
        fn=predict_image,
        inputs=["image", "text"],
        outputs="text",
        live=False,
        title="Dog vs. Cat Classifier"
    )

    # Launch the Gradio interface
    iface.launch()
