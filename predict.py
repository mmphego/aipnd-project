import argparse

from pathlib import Path

import numpy as np
import torch

from torch.autograd import Variable
from torchvision import models

from PIL import Image
from utils import get_names


MEAN = [0.485, 0.456, 0.406]
STD_DEV = [0.229, 0.224, 0.225]


def load_model(trained_model, gpu=False):
    """loads a checkpoint and rebuilds the model"""
    print(f"Loading trained model: {trained_model!r}.")
    checkpoint = torch.load(trained_model)
    device = checkpoint["device"] if not gpu else "cuda"
    arch = checkpoint["arch"]
    if arch.startswith("densenet"):
        arch = "densenet161"
    elif arch.startswith("vgg"):
        arch = "vgg16"
    elif arch.startswith("resnet"):
        arch = "resnet18"

    if not hasattr(models, arch):
        _avail_models = [
            i for i in dir(models) if i.islower() and not i.startswith("_")
        ]
        raise RuntimeError(
            f"Model {arch!r} does not exist, choose from this list: {_avail_models}"
        )
    print(f"Successfully loaded the trained model.")
    print("#" * 80)
    print("Model Specification:", end="\n")
    print(f"Model arch: {arch} and H/W device: {device}.", end="\n\n")
    _model = getattr(models, arch)
    model = _model(pretrained=True)
    assert arch.startswith(model.__module__.split(".")[-1])

    model.to(device)
    model.accuracy = checkpoint.get("accuracy")
    model.arch = checkpoint.get("arch")
    model.class_to_idx = checkpoint.get("class_to_idx")
    model.classifier = checkpoint.get("classifier")
    model.device = checkpoint.get("device")
    model.hidden_layer = checkpoint.get("hidden_layer")
    model.learning_rate = checkpoint.get("learning_rate")
    model.load_state_dict(checkpoint.get("state_dict"))
    return model


def process_image(image_path, size=(256, 256), crop_size=244):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    """
    image_path = Path(image_path)
    if not Path(image_path).is_file():
        raise RuntimeError(f"Image file {image_path.as_posix()!r} doesn't exist.")

    image = Image.open(image_path)
    # https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.resize
    image = image.resize(size, Image.ANTIALIAS)
    assert image.size == size, f"Image resized to {image.size}, instead of {size}"
    h, w = size
    dim = {
        "left": (w - crop_size) / 2,
        "lower": (h - crop_size) / 2,
        "right": ((w - crop_size) / 2) + crop_size,
        "top": ((h - crop_size) / 2) + crop_size,
    }

    cropped_image = image.crop(tuple(dim.values()))
    assert cropped_image.size == (
        crop_size,
        crop_size,
    ), f"Image resized to {cropped_image.size}, instead of {crop_size}"

    mean = np.array(MEAN)
    std_dev = np.array(STD_DEV)

    # make image values to be 1's and 0's
    image_array = np.array(cropped_image) / (size[0] - 1)
    image_array = (image_array - mean) / std_dev
    image_array = image_array.transpose((2, 0, 1))
    assert isinstance(image_array, np.ndarray)
    return image_array


def predict(image, model, gpu=False, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    try:
        assert isinstance(
            image, np.ndarray
        ), "Image is not a numpy array, will process image into an array."
    except AssertionError:
        image = process_image(image)

    model.eval()
    device = model.device
    model.to(device)

    # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type
    # (torch.cuda.FloatTensor) should be the same
    img = (
        torch.from_numpy(image).type(torch.cuda.FloatTensor)
        if gpu
        else torch.from_numpy(image).type(torch.FloatTensor)
    )
    # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    img = img.unsqueeze(0)
    img.to(device)
    img = Variable(img).to(device)

    output = model.forward(img)
    probabilities = torch.exp(output)
    probs, indices = probabilities.topk(topk)

    # Convert cuda tensor to numpy array then to list.
    def tensor_to_list(l):
        return l.to("cpu").detach().numpy().tolist()

    # Convert nested list into a list
    def flatten(l):
        return [item for sublist in l for item in sublist]

    probs = tensor_to_list(probs)
    indices = tensor_to_list(indices)

    probs = flatten(probs) if any(isinstance(i, list) for i in probs) else probs
    indices = flatten(indices) if any(isinstance(i, list) for i in indices) else indices

    class_to_idx = model.class_to_idx
    idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
    classes = [idx_to_class[idx] for idx in indices]

    return probs, classes


def sanity_check(model, *args, **kwargs):
    """Checks if the prediction matched the correct image name."""
    (args,) = args
    if not isinstance(args.image_path, Path):
        image_path = Path(args.image_path)
    image = process_image(args.image_path)
    probs, classes = predict(image, model, gpu=args.gpu, topk=args.topk)

    labels = get_names(args.data_labels)
    predicted_image = labels[image_path.parent.name]
    other_images = [labels[x] for x in classes]
    print("-" * 80)
    print(f"Predication probabilities: {probs}")
    print(f"Prediction classes: {classes}")
    print("-" * 80)
    print(
        f"Predicted the image to be {predicted_image.title()!r} at "
        f"{model.accuracy:.2f} % accuracy."
    )


def main():
    parser = argparse.ArgumentParser("An AI Flower predictor.")

    parser.add_argument(
        "--trained_model",
        default="checkpoint.pth",
        help="Trained model file (.pth) [Default: checkpoint.pth]",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Display top_k probabilities and classes."
    )

    parser.add_argument(
        "--image_path", required=True, help="Path to image",
    )
    parser.add_argument(
        "--data_labels",
        default="cat_to_name.json",
        required=True,
        help="Labels of the data files. [Default: cat_to_name.json]",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU if its available. [Default: False]"
    )

    args = parser.parse_args()

    trained_model = load_model(args.trained_model)
    if trained_model:
        # image_array = process_image(image_path=args.image_path)
        # probs, classes = predict(image_array, trained_model, args)
        sanity_check(trained_model, args)


if __name__ == "__main__":
    main()
