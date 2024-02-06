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
from main import *
import pickle


def get_data_loader():
    # Define the transformation to be applied to each image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
            transforms.ToTensor(),
        ]
    )

    # Load the saved datasets
    train_dataset = torch.load("./preprocessed-data/train_dataset.pth")
    val_dataset = torch.load("./preprocessed-data/val_dataset.pth")

    # Apply the transformation to the datasets
    train_dataset.transform = transform
    val_dataset.transform = transform

    # Create DataLoader for training and validation sets
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_final_conv_size(model, sample_input):
    with torch.no_grad():
        features = model(sample_input)

    return features.view(features.size(0), -1).shape[1]


def extract_features_resnet(model, loader):
    features = []
    labels = []

    # Create a separate instance of ResNet for feature extraction
    resnet18 = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(resnet18.children())[:-1])

    # Create a new model for feature extraction
    feature_extraction_model = nn.Sequential(
        model,
    )

    # feature_extraction_model.eval()

    with torch.no_grad():
        for data in loader:
            images, targets = data
            outputs = feature_extraction_model(images)
            features.append(outputs.squeeze().cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels)
    return features, labels


def get_flattened_size(model, sample_input):
    with torch.no_grad():
        features = model(sample_input)
    return features.view(features.size(0), -1).shape[1]


def extract_features_block1(model, loader):
    features = []
    labels = []

    # Create a separate instance of ResNet for feature extraction from block 1
    resnet18 = models.resnet18(pretrained=True)
    block1 = nn.Sequential(*list(resnet18.children())[:-5])

    # Create a new model for feature extraction from block 1
    feature_extraction_model = nn.Sequential(block1)

    feature_extraction_model.eval()

    with torch.no_grad():
        for data in loader:
            images, targets = data
            print(images.shape)
            outputs = feature_extraction_model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels)
    return features, labels


def extract_features_block3(model, loader):
    features = []
    labels = []

    # Create a separate instance of ResNet for feature extraction from block 3
    resnet18 = models.resnet18(pretrained=True)
    block3 = nn.Sequential(*list(resnet18.children())[:-3])

    # Create a new model for feature extraction from block 3
    feature_extraction_model = nn.Sequential(block3)

    feature_extraction_model.eval()

    with torch.no_grad():
        for data in loader:
            images, targets = data
            outputs = feature_extraction_model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels)
    return features, labels


import torchvision.models as models


def extract_features(model_type, loader):
    features = []
    labels = []

    # Select the model based on the specified type
    if model_type == "resnet":
        model = models.resnet18(pretrained=True)
    elif model_type == "googlenet":
        model = models.inception_v3(pretrained=True, aux_logits=False)
    elif model_type == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError("Unsupported model type")

    # Remove the last fully connected layer (classification layer)
    model = nn.Sequential(*list(model.children())[:-1])

    # Create a new model for feature extraction
    feature_extraction_model = nn.Sequential(model, nn.AdaptiveAvgPool2d(1))

    feature_extraction_model.eval()

    with torch.no_grad():
        for data in loader:
            images, targets = data
            outputs = feature_extraction_model(images)
            features.append(outputs.squeeze().cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels)
    return features, labels
