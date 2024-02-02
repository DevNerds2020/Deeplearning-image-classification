import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
)
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.preprocessing import StandardScaler


# Define the dataset path
dataset_path = "images"

# Define the transformation to be applied to each image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            file
            for file in os.listdir(root_dir)
            if file.lower().endswith((".jpg", ".jpeg"))
        ]

        # Collect unique labels
        self.classes = set()
        for file in self.image_files:
            is_cat = file[0].isupper()
            breed = file.split("_")[0].lower() if not is_cat else file.split("_")[0]
            self.classes.add(breed)

        # Convert set to a sorted list for consistent order
        self.classes = sorted(list(self.classes))

        # Create a dictionary to map class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # determine if it's a cat or dog based on the first letter
        is_cat = image_file[0].isupper()
        breed = (
            image_file.split("_")[0].lower() if not is_cat else image_file.split("_")[0]
        )

        # Map class name to index
        label = self.class_to_idx[breed]

        return image, label


def get_image_features(input_image):
    # Define the transformation to be applied to the image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Load the pre-trained ResNet model with AvgPool layer
    resnet_model = models.resnet18(pretrained=True)
    inception_model = models.googlenet(pretrained=True)
    mobilenet_model = models.mobilenet_v2(pretrained=True)

    avgpool_layer = nn.Sequential(
        *list(resnet_model.children())[:-2]
    )  # Remove the final fully connected layer
    avgpool_layer_inception = nn.Sequential(
        *list(inception_model.children())[:-1]
    )  # Remove the final fully connected layer
    avgpool_layer_mobilenet = nn.Sequential(
        *list(mobilenet_model.children())[:-1]
    )  # Remove the final fully connected layer

    # Set the model to evaluation mode
    avgpool_layer.eval()
    avgpool_layer_inception.eval()
    avgpool_layer_mobilenet.eval()

    input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(
        0
    )  # Add a batch dimension

    # Forward pass to extract features
    with torch.no_grad():
        features = avgpool_layer(input_tensor)
        features_mobilenet = avgpool_layer_mobilenet(input_tensor)
        features_inception = avgpool_layer_inception(input_tensor)

    # Convert the features to a NumPy array
    features_np = features.squeeze().cpu().numpy()
    features_mobilenet_np = features_mobilenet.squeeze().cpu().numpy()
    features_inception_np = features_inception.squeeze().cpu().numpy()

    print(features_np.shape)
    print(features_mobilenet_np.shape)
    print(features_inception_np.shape)

    # Flatten the features
    features_np = features_np.flatten()
    features_mobilenet_np = features_mobilenet_np.flatten()
    features_inception_np = features_inception_np.flatten()

    # Concatenate the features
    image_features = np.concatenate(
        (features_np, features_mobilenet_np, features_inception_np)
    )

    return image_features
