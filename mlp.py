import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import joblib

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

class ImagePreprocessor:
    @staticmethod
    def convertAndNormalize(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return equalized / 255.0

    @staticmethod
    def preprocessImages(images):
        processed = [ImagePreprocessor.convertAndNormalize(img) for img in images]
        return np.array(processed)

class ModelTrainer:
    def __init__(self, hidden_layer_sizes=(500,), learning_rate_init=0.001, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                   learning_rate_init=self.learning_rate_init,
                                   max_iter=self.max_iter)

    def train(self, X_train, y_train):
        print("Training the MLP model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred

    def saveModel(self, path="mlp_model.joblib"):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

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
    
    # решэйп картинок
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # обучение модели
    trainer = ModelTrainer(hidden_layer_sizes=(500,), learning_rate_init=config["learning_rate"], max_iter=200)
    trainer.train(X_train_flat, y_train)
    
    test_accuracy, y_pred = trainer.evaluate(X_test_flat, y_test)
    print(f"Test Accuracy: {test_accuracy}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    class_names = load_class_names(config["labels_file"])
    plot_confusion_matrix(conf_matrix_percent, class_names)
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    trainer.saveModel()

if __name__ == "__main__":
    main(config)
