import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


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


# def show_images(data_path: str, label: str) -> None:
#     images_lst = random_images(data_path)

#     num_rows = 4
#     num_cols = 4
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

#     for i, image_path in enumerate(images_lst):
#         image = cv.imread(image_path)
#         axes[i // num_cols, i % num_cols].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB), interpolation='none')
#         axes[i // num_cols, i % num_cols].set_title(f"{label} Image")
#         axes[i // num_cols, i % num_cols].axis('off')
#     plt.tight_layout()
#     plt.show()


def image_flatten(image: np.array, image_size: int, augmentation: bool = False) -> np.array:
    image = cv.resize(image, (image_size, image_size))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalized_image = cv.equalizeHist(image)

    if augmentation:
        # Apply data augmentation techniques here
        # For example, you can randomly flip the image horizontally
        if random.random() > 0.5:
            equalized_image = cv.flip(equalized_image, 1)

        # You can add more augmentation techniques like rotation, brightness changes, etc.

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


if __name__ == "__main__":
    # # Dataset visual;
    # show_images('D:/Thực tập/Source/Data source/small_dog_cat_dataset-master/small_dog_cat_dataset-master/train/dogs', "Dog")
    # show_images('D:/Thực tập/Source/Data source/small_dog_cat_dataset-master/small_dog_cat_dataset-master/train/cats', "Cat")

    # Create dataset;
    X_train, y_train = create_dataset('D:/Thực tập/Source/Data source/small_dog_cat_dataset-master/small_dog_cat_dataset-master/train', 128)
    print("X_train shape:", X_train.shape)
    print("y_train shape:" ,y_train.shape)
    X_test, y_test = create_dataset('D:/Thực tập/Source/Data source/small_dog_cat_dataset-master/small_dog_cat_dataset-master/test', 128)
    print("X_test shape: ", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Model train;
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Model test;
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuray {accuracy}")