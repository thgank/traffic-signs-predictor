import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
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
    "hog_orientations": 9,
    "hog_pixels_per_cell": (8, 8),
    "hog_cells_per_block": (2, 2),
    "n_estimators": 100,
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
        return gray

    @staticmethod
    def extractHOGFeatures(images, config):
        hog_features = []
        for img in images:
            img = ImagePreprocessor.convertAndNormalize(img)
            feature = hog(
                img,
                orientations=config["hog_orientations"],
                pixels_per_cell=config["hog_pixels_per_cell"],
                cells_per_block=config["hog_cells_per_block"],
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )
            hog_features.append(feature)
        return np.array(hog_features)

class ModelTrainer:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)

    def train(self, X_train, y_train):
        print("Training the Random Forest model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred

    def saveModel(self, path="random_forest_model.joblib"):
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
    
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=config["test_ratio"], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=config["validation_ratio"], random_state=42)
    
    print("Extracting HOG features from training data...")
    X_train_hog = ImagePreprocessor.extractHOGFeatures(X_train, config)
    print("Extracting HOG features from validation data...")
    X_val_hog = ImagePreprocessor.extractHOGFeatures(X_val, config)
    print("Extracting HOG features from test data...")
    X_test_hog = ImagePreprocessor.extractHOGFeatures(X_test, config)
    
    trainer = ModelTrainer(n_estimators=config["n_estimators"])
    trainer.train(X_train_hog, y_train)
    
    test_accuracy, y_pred = trainer.evaluate(X_test_hog, y_test)
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
