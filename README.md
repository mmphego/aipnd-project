# AI Programming with Python Project

**Note:** This is the final Project of the Udacity AI with Python Nanodegree

Project code for Udacity's AI Programming with Python Nanodegree program.
In this project, I had to develop code for an image classifier built with PyTorch, then convert it into a command line application for portability.

# Prerequisites

Run the command below in order to install the dependencies.

```bash
python3.6 -m pip install -r requirements.txt
```


# Core Curriculum

This section consists of all the lessons and projects I needed to complete in order to receive a Udacity Nanodegree certificate.

- [x] Part 1: Introduction to AI Programming
Welcome to the AI programming with python Nanodegree Program! Come and explore the beautiful world of AI.
- [x] Part 2: Intro to Python
Learn Python, one of the most widely used programming languages in the industry, particularly in AI.
- [x] Part 3: Numpy, Pandas, Matplotlib
Let's focus on library packages for Python, such as : Numpy (which adds support for large data), Pandas (which is used for data manipulation and analysis) And Matplotlib (which is used for data visualization).
- [x] Part 4: Linear Algebra Essentials
Learn the basics of the beautiful world of Linear Algebra and why it is such an important mathematical tool in the world of AI.
- [x] Part 5: Neural Networks
Acquire a solid foundation in deep learning and neural networks. Learn about techniques for how to improve the training of a neural network, and how to use PyTorch for building deep learning models.
- [x] Part 6: Image Classifier Project
In the final project for this course, you'll build a state-of-the-art image classification application.
Project: Create Your Own Image Classifier

# Checklist/ToDo
## Developing an Image Classifier with Deep Learning on Jupyter Notebook

In this first part of the project, I worked through a Jupyter notebook to implement an image classifier with [PyTorch](https://pytorch.org/).  [Udacity](https://www.udacity.com/) provided a workspace with a GPU for working on the project.
## Part 1 - Development Notebook [TODO]
- [x] **Package Imports**: All the necessary packages and modules are imported in the first cell of the notebook.
- [x] **Training data augmentation :** `torchvision` transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping.
- [x] **Data normalization:** The training, validation, and testing data is appropriately cropped and normalized.
- [x] **Data loading:** The data for each set (train, validation, test) is loaded with `torchvision`'s ImageFolder.
- [x] **Data batching:** The data for each set is loaded with `torchvision`'s `DataLoader`.
- [x] **Pretrained Network:** A pretrained network such as VGG16 is loaded from `torchvision.models` and the parameters are frozen.
- [x] **Feedforward Classifier:** A new `feedforward` network is defined for use as a classifier using the features as input.
- [x] **Training the network:** The parameters of the `feedforward` classifier are appropriately trained, while the parameters of the feature network are left static.
- [x] **Validation Loss and Accuracy:** During training, the validation loss and accuracy are displayed.
- [x] **Testing Accuracy:** The network's accuracy is measured on the test data.
- [x] **Saving the model:** The trained model is saved as a checkpoint along with associated hyperparameters and the `class_to_idx` dictionary.
- [x] **Loading checkpoints:** There is a function that successfully loads a checkpoint and rebuilds the model.
- [x] **Image Processing:** The `process_image` function successfully converts a PIL  image into an object that can be used as input to a trained model
- [x] **Class Prediction:** The `predict` function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image.
- [x] **Sanity Checking with matplotlib:** A `matplotlib` figure is created displaying an image and its associated top 5 most probable classes with actual flower names.

## Building the command line application

After I had built and trained a deep neural network on the flower data set, it was time to convert it into an application that others can use. My application consists of two(2) Python scripts that run from the command line. For testing, I added the `checkpoint` saved in the first part.

### Part 2 - Command Line Application [TODO]

- [x] **Training a network:** `train.py` successfully trains a new network on a dataset of images.
- [x] **Training validation log:** The training loss, validation loss, and validation accuracy are printed out as a network trains.
- [x] **Model architecture:** The training script allows users to choose from at least two different architectures available from `torchvision.models`.
- [x] **Model hyperparameters:** The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs.
- [x] **Training with GPU:** The training script allows users to choose training the model on a GPU.
- [x] **Predicting classes:** The `predict.py` script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability.
- [x] **Top K classes:** The `predict.py` script allows users to print out the top K classes along with associated probabilities.
- [x] **Displaying class names:** The `predict.py` script allows users to load a JSON file that maps the class values to other category names.
- [x] **Predicting with GPU:** The `predict.py` script allows users to use the GPU to calculate the predictions.


#### Specifications

The project includes at 3(three) files `train.py`,  `predict.py` and `utils.py`. The first file, `train.py`, will train a new network on a dataset and save the model as a `checkpoint`. The second file, `predict.py`, uses a trained network to predict the class for an input image. The third file, `utils.py` contains helper and utility functions.

*   Train a new network on a data set with `train.py`
    *   Basic usage: `python train.py data_directory`
    *   Prints out training loss, validation loss, and validation accuracy as the network trains
    *   Options:
        *   Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
        *   Choose architecture: `python train.py data_dir --arch "vgg13"`
        *   Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
        *   Use GPU for training: `python train.py data_dir --gpu`

**Example:**
```bash
python3.6 train.py --gpu --epoch 2 --arch vgg16
```

*   Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.
    *   Basic usage: `python predict.py /path/to/image checkpoint`
    *   Options:
        *   Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
        *   Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
        *   Use GPU for inference: `python predict.py input checkpoint --gpu`
```bash
python3.6 predict.py --image_path flowers/test/1/image_06752.jpg --data_labels cat_to_name.json --gpu
```

Model checkpoint: https://www.dropbox.com/s/bmd2e2l4s93kvgg/checkpoint.pth?dl=0

Download image dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

Structure the directory as follows to avoid disappointment.:
```bash
$ tree flowers -L 3

flowers
├── test
│   ├── 1
│   │   ├── image_06743.jpg
│   │   └── image_06764.jpg
│   ├── 102
│   │   ├── image_08004.jpg
│   │   ├── image_08012.jpg
│   │   ├── image_08015.jpg
│   │   ├── image_08023.jpg
│   │   ├── image_08030.jpg
│   │   └── image_08042.jpg
├── train
│   ├── 1
│   │   ├── image_07086.jpg
│   │   ├── image_07087.jpg
│   ├── 102
│   │   ├── image_08000.jpg
│   │   ├── image_08001.jpg
│   │   ├── image_08003.jpg
└── valid
    ├── 1
    │   ├── image_06739.jpg
    │   ├── image_06749.jpg
    │   ├── image_06755.jpg
    ├── 102
    │   ├── image_07895.jpg
    │   ├── image_07904.jpg
    │   ├── image_07905.jpg

270 directories, 5715 files
```

Further reading: [Automated Flower Classification over a Large Number of Classes by Maria-Elena Nilsback and Andrew Zisserman](https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/)
