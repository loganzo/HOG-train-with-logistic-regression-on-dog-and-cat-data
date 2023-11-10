import os
import random
import cv2 as cv
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gradio as gr

def random_image_index(lenght_data: int) -> int:
    return np.random.randint(0, lenght_data - 1)

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

def X_y_data(data: list) -> list:
    X = []
    y = []

    for i in data:
        X.append(i[0])
        y.append(i[1])

    return X, y

def create_dataset(data_path: str, image_size: int) -> list:
    data_set = []

    for label in os.listdir(data_path):
        label_folder = os.path.join(data_path, label)

        for image_name in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_name)
            image = cv.imread(image_path)

            if image is not None:
                image = image_flatten(image, image_size)
                if label == "dogs":
                    data_set.append((image, 1))
                else:
                    data_set.append((image, 0))
            else:
                print(f"Can't open image {image_path}")

    random.shuffle(data_set)
    X, y = X_y_data(data_set)
    X = np.array(X)
    y = np.array(y)

    return X, y

# Gradio interface function
def predict_image(img):
    img_features = image_flatten(img, 128)
    prediction = model.predict([img_features])[0]
    return "Dog" if prediction == 1 else "Cat"

# Main function
if __name__ == "__main__":
    # Create dataset
    X_train, y_train = create_dataset('D:/try hard/Source/Data source/small_dog_cat_dataset-master/small_dog_cat_dataset-master/train', 128)
    X_test, y_test = create_dataset('D:/try hard/Source/Data source/small_dog_cat_dataset-master/small_dog_cat_dataset-master/test', 128)

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
        inputs="image",
        outputs="text",
        live=True
    )

    # Launch the Gradio interface
    iface.launch()
