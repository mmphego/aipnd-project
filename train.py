#!/usr/bin/env python
# coding: utf-8

import argparse
import time

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch  # version 0.4.0

from matplotlib import pyplot as plt

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from utils import get_names

IMG_SIZE = 224
IMG_ROTATION = 30
ORIG_IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD_DEV = [0.229, 0.224, 0.225]


def build_training_data(*args, **kwargs):
    """Transforms for the training, validation, and testing sets."""
    args = args[0]
    if not Path(args.data_dir).exists():
        raise RuntimeError(f"Expected directory {args.data_dir!r} doesn't exist.")

    train_dir = Path(args.data_dir) / "train"
    valid_dir = Path(args.data_dir) / "valid"
    test_dir = Path(args.data_dir) / "test"

    if not (train_dir or valid_dir or test_dir):
        raise RuntimeError("Missing data directories.")

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(IMG_ROTATION),
            transforms.RandomResizedCrop(
                IMG_SIZE
            ),  # Randomly resized and cropped images to 224x224
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # Convert to a tensor
            transforms.Normalize(  # Normalize
                MEAN, STD_DEV,  # Means  # Standard deviation
            ),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(ORIG_IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),  # Convert to a tensor
            transforms.Normalize(  # Normalize
                MEAN, STD_DEV,  # Means  # Standard deviation
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(ORIG_IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),  # Convert to a tensor
            transforms.Normalize(  # Normalize
                MEAN, STD_DEV,  # Means  # Standard deviation
            ),
        ]
    )

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    _datasets = {
        "train_datasets": train_datasets,
        "valid_datasets": valid_datasets,
        "test_datasets": test_datasets,
    }

    # Using the image datasets and the transforms, define the dataloaders
    train_loaders = DataLoader(train_datasets, batch_size=32, shuffle=True)
    valid_loaders = DataLoader(valid_datasets, batch_size=32, shuffle=True)
    test_loaders = DataLoader(test_datasets, batch_size=32, shuffle=True)

    _dataloaders = {
        "train_loaders": train_loaders,
        "valid_loaders": valid_loaders,
        "test_loaders": test_loaders,
    }

    return {"datasets": _datasets, "dataloaders": _dataloaders}

def train_model(*args, **kwargs):
    """Building and training the classifier."""
    args = args[0]

    _avail_models = [i for i in dir(models) if i.islower() and not i.startswith("_")]
    # Load a pre-trained network
    if not hasattr(models, args.arch):
        raise RuntimeError(
            f"Model {args.arch!r} does not exist, choose from this list: {_avail_models}"
        )

    _model = getattr(models, args.arch)
    model = _model(pretrained=True)
    model.arch = model.__name__ = model.__module__.split(".")[-1]
    print("#" * 80)
    print("Model Specifications.\n")
    print(
        f"{model.arch.title()!r} model Before classifier update: \n>> {model.classifier}",
        end="\n\n",
    )

    def new_classifier(model):
        """Define a new, untrained feed-forward network as a classifier, using
        ReLU activations and dropout.
        """
        for param in model.parameters():
            param.requires_grad = False

        in_features = (
            model.classifier.in_features
            if hasattr(model.classifier, "in_features")
            else model.classifier[0].in_features
        )
        out_features = (
            args.hidden_layer
            if hasattr(model.classifier, "out_features")
            else model.classifier[0].out_features
        )
        data_labels = get_names(args.data_labels)
        new_classifier = nn.Sequential(
            OrderedDict(
                [
                    ("drop_out", nn.Dropout(p=0.5)),
                    ("fc1", nn.Linear(in_features, out_features)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("drop_out", nn.Dropout(p=0.5)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(out_features, len(data_labels))),
                    ("relu", nn.ReLU()),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
        return new_classifier

    model.classifier = new_classifier(model)
    print("->" * 40, end="\n\n")
    print(
        f"{model.arch.title()!r} model After classifier update: \n>> {model.classifier}"
    )
    print("#" * 80)
    device = "cuda" if args.gpu else "cpu"
    model.device = device
    if torch.cuda.is_available():
        print("Note: Model is being trained using GPU which will speed up training.")

    device = torch.device(device)
    model.to(device)
    print(f"Model using the {device!r} device for training.")

    # Train the classifier layers using backpropagation using the pre-trained
    # network to get the features
    learning_rate = args.learning_rate
    epochs = args.epochs
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=model.classifier.parameters(), lr=learning_rate)
    print(
        f"Model to be trained at: epochs: {epochs}, learning rate = {learning_rate!r}, "
        f"criterion = {criterion.__module__.split('.')[-1]!r} and, "
        f"optimizer = {optimizer.__module__.split('.')[-1]!r}"
    )

    def validation(model, validloader, criterion):
        test_loss = 0
        accuracy = 0
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            probability = torch.exp(output)
            equality = labels.data == probability.max(dim=1)[1]
            accuracy += equality.type(torch.FloatTensor).mean()
        return test_loss, accuracy

    def train(print_every=40, epochs=5, **kwargs):
        train_loaders = kwargs["dataloaders"]["train_loaders"]
        valid_loaders = kwargs["dataloaders"]["valid_loaders"]

        steps = 0
        running_loss = 0
        for epoch in range(epochs):
            model.train()
            running_loss = 0
            for images, labels in iter(train_loaders):
                images, labels = images.to(device), labels.to(device)
                steps += 1

                optimizer.zero_grad()

                output = model.forward(images)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        validation_loss, validation_accuracy = validation(
                            model, valid_loaders, criterion
                        )

                    print(
                        f"Timestamp: {datetime.now()}.."
                        f"Model: {model.arch}.."
                        f"Epoch: {epoch + 1} / {epochs}.."
                        f"Training Loss: {running_loss / print_every:.3f}%.."
                        f"Validation Loss: {validation_loss / len(valid_loaders):.3f}%.."
                        f"Validation Accuracy: {validation_accuracy / len(valid_loaders) * 100:.3f}%.."
                    )
                    running_loss = 0
                    model.train()

        return model

    print("Training started...", end="\n\n")
    start_time = time.time()
    trained_model = train(epochs=epochs, **kwargs)
    end_time = time.time()

    print(f"\nModel took {(end_time - start_time)/3600:.2f} hour(s) to train...")
    return trained_model


def test_model(model, **kwargs):
    test_loaders = kwargs["dataloaders"]["test_loaders"]
    device = model.device
    accuracy = 0
    model.to(device)
    model.eval()

    for images, labels in iter(test_loaders):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            output = model.forward(images)
            probability = torch.exp(output)

            equality = labels.data == probability.max(dim=1)[1]
            accuracy += equality.type(torch.FloatTensor).mean()

    model_accuracy = (accuracy / len(test_loaders)) * 100
    model.accuracy = model_accuracy

    if model_accuracy <= 70:
        print("Model failed to meet minimum accuracy of 70%")
    else:
        print(f"Model Accuracy after testing is {model_accuracy:.2f} %")
        return model_accuracy


def save_checkpoint(model, accuracy, *args, **kwargs):
    """After model is trained, save it for future use."""
    args, = args
    model.class_to_idx = kwargs["datasets"]["train_datasets"].class_to_idx

    checkpoint = {
        "accuracy": model.accuracy,
        "arch": model.arch,
        "class_to_idx": model.class_to_idx,
        "classifier": model.classifier,
        "device": model.device,
        "hidden_layer": args.hidden_layer,
        "learning_rate": args.learning_rate,
        "n_inputs": model.classifier.fc2.in_features,
        "n_output": model.classifier.fc2.out_features,
        "state_dict": model.state_dict(),
    }

    checkpoint_path = Path(args.save_dir) / Path(args.checkpoint)
    torch.save(checkpoint, checkpoint_path)
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Failed to save the 'checkpoint.pth' into {checkpoint_path.as_posix()!r}"
        )
    print(f"Saved the trained model: {checkpoint_path.as_posix()!r}")


def main():
    parser = argparse.ArgumentParser(
        "An AI Flower Classification over a Large Number of Classes."
    )
    parser.add_argument(
        "--data_dir",
        nargs=1,
        default="flowers",
        help="Directory of the data. [Default: flowers]",
    )
    parser.add_argument(
        "--data_labels",
        default="cat_to_name.json",
        help="Labels of the data files. [Default: cat_to_name.json]",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint.pth",
        help="Trained model file (.pth) [Default: checkpoint.pth]",
    )
    parser.add_argument(
        "--save_dir",
        default=".",
        help="Directory for saving the checkpoint. [Default: cwd]",
    )
    parser.add_argument(
        "--arch",
        default="densenet161",
        help="The CNN model architecture to be used for training. [Default: densenet161]",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Rate at which network learns. [Default: 0.001]",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="No. of Epochs. [Default: 5]"
    )
    parser.add_argument(
        "--hidden_layer",
        type=int,
        default=1000,
        help="No. of perceptrons in the Hidden Layer. [Default: densenet161 = 1000]",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU if its available. [Default: False]"
    )

    args, opts = parser.parse_known_args()
    if opts:
        if len(opts) > 1:
            raise RuntimeError(
                "Expected only 1 positional argument which should be dir containing the "
                "images."
            )
        args.data_dir = opts[0]

    training_data = build_training_data(args)
    trained_model = train_model(args, **training_data)
    model_accuracy = test_model(trained_model, **training_data)
    if model_accuracy:
        save_checkpoint(trained_model, model_accuracy, args, **training_data)


if __name__ == "__main__":
    main()
