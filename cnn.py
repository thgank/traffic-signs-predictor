import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.optimizers import Adam

config = {
    "dataset_path": "Dataset",
    "labels_file": "labels.csv",
    "batch_size": 32,
    "epochs": 30,
    "image_dims": (32, 32, 3),
    "test_ratio": 0.2,
    "validation_ratio": 0.2,
    "learning_rate": 0.001,
}

# сгрузка датасета
class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def loadImagesAndLabels(self):
        images, labels = [], []
        classes = sorted(os.listdir(self.dataset_path), key=lambda x: int(x))

        for idx, className in enumerate(classes):
            folderPath = os.path.join(self.dataset_path, className)
            for imgFile in os.listdir(folderPath):
                imgPath = os.path.join(folderPath, imgFile)
                img = cv2.imread(imgPath)
                if img is not None:
                    images.append(img)
                    labels.append(idx)
                else:
                    print(f"Warning: Failed to load {imgPath}")
        
        print(f"Total images: {len(images)} across {len(classes)} classes.")
        return np.array(images), np.array(labels), len(classes)

# подготовка картинок для обучения
class ImagePreprocessor:
    @staticmethod
    def convertAndNormalize(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return equalized / 255.0

    @staticmethod
    def preprocessImages(images):
        processed = [ImagePreprocessor.convertAndNormalize(img) for img in images]
        return np.expand_dims(processed, -1)

# обучение модели в 30 поколений
class ModelTrainer:
    def __init__(self, input_shape, num_classes, learning_rate):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._buildModel()

    def _buildModel(self):
        model = Sequential([
            RandomRotation(0.1, input_shape=self.input_shape),
            RandomZoom(0.2),
            RandomTranslation(0.1, 0.1),
            Conv2D(60, (5, 5), activation='relu'),
            Conv2D(60, (5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(30, (3, 3), activation='relu'),
            Conv2D(30, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(500, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size, epochs):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def saveModel(self, path="cnn_model.keras"):
        self.model.save(path)

# построение графика корреляции
def plot_confusion_matrix(conf_matrix, class_names):
    class_ids = [f"{i}" for i in range(len(class_names))] 
    plt.figure(figsize=(35, 25))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_ids, yticklabels=class_ids, 
                annot_kws={"size": 8},
                linewidths=0.5,
                cbar_kws={'label': 'Accuracy Rate (%)'})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Predicted ClassId')
    plt.ylabel('True ClassId')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.show()


def load_class_names(label_file):
    label_df = pd.read_csv(label_file).sort_values('ClassId')
    return label_df['Name'].values

def main(config):
    data_loader = DatasetLoader(config["dataset_path"])
    images, labels, num_classes = data_loader.loadImagesAndLabels()
    
    # разделение данных для обучения
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=config["test_ratio"], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=config["validation_ratio"], random_state=42)
    
    X_train = ImagePreprocessor.preprocessImages(X_train)
    X_val = ImagePreprocessor.preprocessImages(X_val)
    X_test = ImagePreprocessor.preprocessImages(X_test)
    
    # кодировка колонн
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    # обучение модели
    trainer = ModelTrainer(input_shape=(config["image_dims"][0], config["image_dims"][1], 1),
                           num_classes=num_classes, learning_rate=config["learning_rate"])
    history = trainer.train(X_train, y_train, X_val, y_val, config["batch_size"], config["epochs"])
    
    test_score = trainer.evaluate(X_test, y_test)
    print(f"Test Loss: {test_score[0]}, Test Accuracy: {test_score[1]}")
    
    y_pred = np.argmax(trainer.model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    class_names = load_class_names(config["labels_file"])
    plot_confusion_matrix(conf_matrix_percent, class_names)
    
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    trainer.saveModel()

if __name__ == "__main__":
    main(config)
