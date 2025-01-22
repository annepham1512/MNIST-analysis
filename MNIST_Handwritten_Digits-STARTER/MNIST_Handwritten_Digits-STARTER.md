# Introduction

In this project, you will build a neural network of your own design to evaluate the MNIST dataset.

Some of the benchmark results on MNIST include can be found [on Yann LeCun's page](https://webcache.googleusercontent.com/search?q=cache:stAVPik6onEJ:yann.lecun.com/exdb/mnist) and include:

88% [Lecun et al., 1998](https://hal.science/hal-03926082/document)

95.3% [Lecun et al., 1998](https://hal.science/hal-03926082v1/document)

99.65% [Ciresan et al., 2011](http://people.idsia.ch/~juergen/ijcai2011.pdf)


MNIST is a great dataset for sanity checking your models, since the accuracy levels achieved by large convolutional neural networks and small linear models are both quite high. This makes it important to be familiar with the data.

## Installation


```python
# Update the PATH to include the user installation directory. 
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"

# Restart the Kernel before you move on to the next step.
```

#### Important: Restart the Kernel before you move on to the next step.


```python
# Install requirements
!python -m pip install -r requirements.txt
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: opencv-python-headless==4.5.3.56 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (4.5.3.56)
    Requirement already satisfied: matplotlib==3.4.3 in /opt/conda/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (3.4.3)
    Requirement already satisfied: numpy==1.21.2 in /opt/conda/lib/python3.7/site-packages (from -r requirements.txt (line 3)) (1.21.2)
    Requirement already satisfied: pillow==7.0.0 in /opt/conda/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (7.0.0)
    Requirement already satisfied: bokeh==2.1.1 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (2.1.1)
    Requirement already satisfied: torch==1.11.0 in /opt/conda/lib/python3.7/site-packages (from -r requirements.txt (line 6)) (1.11.0)
    Requirement already satisfied: torchvision==0.12.0 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 7)) (0.12.0)
    Requirement already satisfied: tqdm==4.63.0 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 8)) (4.63.0)
    Requirement already satisfied: ipywidgets==7.7.0 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 9)) (7.7.0)
    Requirement already satisfied: livelossplot==0.5.4 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 10)) (0.5.4)
    Requirement already satisfied: pytest==7.1.1 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 11)) (7.1.1)
    Requirement already satisfied: pandas==1.3.5 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 12)) (1.3.5)
    Requirement already satisfied: seaborn==0.11.2 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 13)) (0.11.2)
    Requirement already satisfied: jupyter==1.0.0 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 14)) (1.0.0)
    Requirement already satisfied: ipykernel==4.10.0 in /root/.local/lib/python3.7/site-packages (from -r requirements.txt (line 15)) (4.10.0)
    Requirement already satisfied: pyparsing>=2.2.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib==3.4.3->-r requirements.txt (line 2)) (2.4.6)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib==3.4.3->-r requirements.txt (line 2)) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.7/site-packages (from matplotlib==3.4.3->-r requirements.txt (line 2)) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/lib/python3.7/site-packages (from matplotlib==3.4.3->-r requirements.txt (line 2)) (2.8.1)
    Requirement already satisfied: tornado>=5.1 in /opt/conda/lib/python3.7/site-packages (from bokeh==2.1.1->-r requirements.txt (line 5)) (5.1.1)
    Requirement already satisfied: PyYAML>=3.10 in /opt/conda/lib/python3.7/site-packages (from bokeh==2.1.1->-r requirements.txt (line 5)) (5.3)
    Requirement already satisfied: Jinja2>=2.7 in /opt/conda/lib/python3.7/site-packages (from bokeh==2.1.1->-r requirements.txt (line 5)) (2.11.1)
    Requirement already satisfied: packaging>=16.8 in /opt/conda/lib/python3.7/site-packages (from bokeh==2.1.1->-r requirements.txt (line 5)) (20.1)
    Requirement already satisfied: typing-extensions>=3.7.4 in /opt/conda/lib/python3.7/site-packages (from bokeh==2.1.1->-r requirements.txt (line 5)) (3.7.4.1)
    Requirement already satisfied: requests in /opt/conda/lib/python3.7/site-packages (from torchvision==0.12.0->-r requirements.txt (line 7)) (2.23.0)
    Requirement already satisfied: ipython-genutils~=0.2.0 in /opt/conda/lib/python3.7/site-packages (from ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.2.0)
    Requirement already satisfied: traitlets>=4.3.1 in /opt/conda/lib/python3.7/site-packages (from ipywidgets==7.7.0->-r requirements.txt (line 9)) (4.3.3)
    Requirement already satisfied: nbformat>=4.2.0 in /opt/conda/lib/python3.7/site-packages (from ipywidgets==7.7.0->-r requirements.txt (line 9)) (5.0.4)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0; python_version >= "3.6" in /root/.local/lib/python3.7/site-packages (from ipywidgets==7.7.0->-r requirements.txt (line 9)) (3.0.13)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /root/.local/lib/python3.7/site-packages (from ipywidgets==7.7.0->-r requirements.txt (line 9)) (3.6.9)
    Requirement already satisfied: ipython>=4.0.0; python_version >= "3.3" in /opt/conda/lib/python3.7/site-packages (from ipywidgets==7.7.0->-r requirements.txt (line 9)) (7.13.0)
    Requirement already satisfied: py>=1.8.2 in /root/.local/lib/python3.7/site-packages (from pytest==7.1.1->-r requirements.txt (line 11)) (1.11.0)
    Requirement already satisfied: tomli>=1.0.0 in /root/.local/lib/python3.7/site-packages (from pytest==7.1.1->-r requirements.txt (line 11)) (2.0.1)
    Requirement already satisfied: attrs>=19.2.0 in /opt/conda/lib/python3.7/site-packages (from pytest==7.1.1->-r requirements.txt (line 11)) (19.3.0)
    Requirement already satisfied: importlib-metadata>=0.12; python_version < "3.8" in /opt/conda/lib/python3.7/site-packages (from pytest==7.1.1->-r requirements.txt (line 11)) (1.5.0)
    Requirement already satisfied: pluggy<2.0,>=0.12 in /root/.local/lib/python3.7/site-packages (from pytest==7.1.1->-r requirements.txt (line 11)) (1.2.0)
    Requirement already satisfied: iniconfig in /root/.local/lib/python3.7/site-packages (from pytest==7.1.1->-r requirements.txt (line 11)) (2.0.0)
    Requirement already satisfied: pytz>=2017.3 in /opt/conda/lib/python3.7/site-packages (from pandas==1.3.5->-r requirements.txt (line 12)) (2019.3)
    Requirement already satisfied: scipy>=1.0 in /opt/conda/lib/python3.7/site-packages (from seaborn==0.11.2->-r requirements.txt (line 13)) (1.7.1)
    Requirement already satisfied: jupyter-console in /root/.local/lib/python3.7/site-packages (from jupyter==1.0.0->-r requirements.txt (line 14)) (6.6.3)
    Requirement already satisfied: nbconvert in /opt/conda/lib/python3.7/site-packages (from jupyter==1.0.0->-r requirements.txt (line 14)) (5.6.1)
    Requirement already satisfied: notebook in /opt/conda/lib/python3.7/site-packages (from jupyter==1.0.0->-r requirements.txt (line 14)) (5.7.4)
    Requirement already satisfied: qtconsole in /root/.local/lib/python3.7/site-packages (from jupyter==1.0.0->-r requirements.txt (line 14)) (5.4.4)
    Requirement already satisfied: jupyter-client in /opt/conda/lib/python3.7/site-packages (from ipykernel==4.10.0->-r requirements.txt (line 15)) (6.0.0)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib==3.4.3->-r requirements.txt (line 2)) (45.2.0.post20200209)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from cycler>=0.10->matplotlib==3.4.3->-r requirements.txt (line 2)) (1.16.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/lib/python3.7/site-packages (from Jinja2>=2.7->bokeh==2.1.1->-r requirements.txt (line 5)) (1.1.1)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests->torchvision==0.12.0->-r requirements.txt (line 7)) (1.25.7)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests->torchvision==0.12.0->-r requirements.txt (line 7)) (2.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests->torchvision==0.12.0->-r requirements.txt (line 7)) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests->torchvision==0.12.0->-r requirements.txt (line 7)) (2019.11.28)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from traitlets>=4.3.1->ipywidgets==7.7.0->-r requirements.txt (line 9)) (4.4.2)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /opt/conda/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets==7.7.0->-r requirements.txt (line 9)) (3.2.0)
    Requirement already satisfied: jupyter-core in /opt/conda/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets==7.7.0->-r requirements.txt (line 9)) (4.6.3)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.1.0)
    Requirement already satisfied: pexpect; sys_platform != "win32" in /opt/conda/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (4.8.0)
    Requirement already satisfied: jedi>=0.10 in /opt/conda/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.16.0)
    Requirement already satisfied: pickleshare in /opt/conda/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.7.5)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (2.5.2)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (3.0.3)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/lib/python3.7/site-packages (from importlib-metadata>=0.12; python_version < "3.8"->pytest==7.1.1->-r requirements.txt (line 11)) (3.0.0)
    Requirement already satisfied: pyzmq>=17 in /opt/conda/lib/python3.7/site-packages (from jupyter-console->jupyter==1.0.0->-r requirements.txt (line 14)) (19.0.0)
    Requirement already satisfied: bleach in /opt/conda/lib/python3.7/site-packages (from nbconvert->jupyter==1.0.0->-r requirements.txt (line 14)) (3.1.1)
    Requirement already satisfied: entrypoints>=0.2.2 in /opt/conda/lib/python3.7/site-packages (from nbconvert->jupyter==1.0.0->-r requirements.txt (line 14)) (0.3)
    Requirement already satisfied: mistune<2,>=0.8.1 in /opt/conda/lib/python3.7/site-packages (from nbconvert->jupyter==1.0.0->-r requirements.txt (line 14)) (0.8.4)
    Requirement already satisfied: defusedxml in /opt/conda/lib/python3.7/site-packages (from nbconvert->jupyter==1.0.0->-r requirements.txt (line 14)) (0.6.0)
    Requirement already satisfied: testpath in /opt/conda/lib/python3.7/site-packages (from nbconvert->jupyter==1.0.0->-r requirements.txt (line 14)) (0.4.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /opt/conda/lib/python3.7/site-packages (from nbconvert->jupyter==1.0.0->-r requirements.txt (line 14)) (1.4.2)
    Requirement already satisfied: Send2Trash in /opt/conda/lib/python3.7/site-packages (from notebook->jupyter==1.0.0->-r requirements.txt (line 14)) (1.5.0)
    Requirement already satisfied: prometheus-client in /opt/conda/lib/python3.7/site-packages (from notebook->jupyter==1.0.0->-r requirements.txt (line 14)) (0.7.1)
    Requirement already satisfied: terminado>=0.8.1 in /opt/conda/lib/python3.7/site-packages (from notebook->jupyter==1.0.0->-r requirements.txt (line 14)) (0.8.3)
    Requirement already satisfied: qtpy>=2.4.0 in /root/.local/lib/python3.7/site-packages (from qtconsole->jupyter==1.0.0->-r requirements.txt (line 14)) (2.4.1)
    Requirement already satisfied: pyrsistent>=0.14.0 in /opt/conda/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.15.7)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.7/site-packages (from pexpect; sys_platform != "win32"->ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.6.0)
    Requirement already satisfied: parso>=0.5.2 in /opt/conda/lib/python3.7/site-packages (from jedi>=0.10->ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.6.1)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0; python_version >= "3.3"->ipywidgets==7.7.0->-r requirements.txt (line 9)) (0.1.8)
    Requirement already satisfied: webencodings in /opt/conda/lib/python3.7/site-packages (from bleach->nbconvert->jupyter==1.0.0->-r requirements.txt (line 14)) (0.5.1)


## Imports


```python
## This cell contains the essential imports you will need – DO NOT CHANGE THE CONTENTS! ##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
```


```python
from torchvision.datasets import MNIST
trainset = MNIST(root='./data', train=True, download=True, transform=transforms)
```

## Load the Dataset

Specify your transforms as a list if you intend to .
The transforms module is already loaded as `transforms`.

MNIST is fortunately included in the torchvision module.
Then, you can create your dataset using the `MNIST` object from `torchvision.datasets` ([the documentation is available here](https://pytorch.org/vision/stable/datasets.html#mnist)).
Make sure to specify `download=True`! 

Once your dataset is created, you'll also need to define a `DataLoader` from the `torch.utils.data` module for both the train and the test set.


```python
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Create training set and define training dataloader
mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)

# Create test set and define test dataloader
mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform)

# Create validation set
mnist_valset = MNIST(root='./data', train=False, download=True, transform=transform)

# Define DataLoader for training data
train_loader = DataLoader(
    dataset=mnist_trainset,
    batch_size=64,
    shuffle=True
)

# Define DataLoader for test data
test_loader = DataLoader(
    dataset=mnist_testset,
    batch_size=64,
    shuffle=False
)

# Define DataLoader for validation data
val_loader = DataLoader(
    dataset=mnist_valset,
    batch_size=64,
    shuffle=False
)

```

## Justify your preprocessing

In your own words, why did you choose the transforms you chose? If you didn't use any preprocessing steps, why not?

First, I used `ToTensor()`. This transform converts the images from the dataset (which are in the form of PIL images) into PyTorch tensors. PyTorch operaters with tensors, which are multidimensional arrays similar to NumPy arrays, and this conversion is essential for the data to be used in the neural network. Additionally, this step scalses the pixel values from the range [0,255] to [0,1] making the values more manageable for the model.

The next step is to normalize the data, in which I used `Normalize((0.5,), (0.5,))` for. This transform normalizes the image pixel values to fall within the range [-1,1] with a mean of 0.5 and a standard deviation of 0.5 for the single grayscale channel. The original MNIST dataset has pixel values in the range [0,1] after applying `ToTensor()`, and this normalize them to more center range around 0. This process is necessary since it helps improve the training process by ensuring that the input data us centered around 0 and has a standard deviation of 1. Neural network tends to perform better and converge faster when input values are normalized. It also helps avoid issues like vanishing or exploding gradients during backpropagation, which can occur if the input data has vary large or very small values.

So why these specific transforms? I suppose that the combination of converting images to tensors and normalizing the data ensure that the neural network receives the data in an optimal format for learning. Without these steps, the pixel values would be too large since they are in the range of [0,225] without normalization, which would slow down or hinder the network's ability to learn effectively.

## Explore the Dataset
Using matplotlib, numpy, and torch, explore the dimensions of your data.

You can view images using the `show5` function defined below – it takes a data loader as an argument.
Remember that normalized images will look really weird to you! You may want to try changing your transforms to view images.
Typically using no transforms other than `toTensor()` works well for viewing – but not as well for training your network.
If `show5` doesn't work, go back and check your code for creating your data loaders and your training/test sets.


```python
## This cell contains a function for showing 5 images from a dataloader – DO NOT CHANGE THE CONTENTS! ##
def show5(img_loader):
    data_iter = iter(img_loader)  # Create an iterator from the DataLoader
    images, labels = data_iter.next()  # Get the first batch of images and labels
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))  # Create a figure with 5 subplots
    
    for i in range(5):
        # Move tensor to CPU and convert to NumPy array for displaying
        img = images[i].numpy().squeeze()  # Remove unnecessary channel dimension
        
        axes[i].imshow(img, cmap='gray')  # Show the image in grayscale
        axes[i].set_title(f"Label: {labels[i].item()}")  # Set the title as the label
        axes[i].axis('off')  # Turn off the axis
    
    plt.show()  # Display the plot
```


```python
# Display 5 sample images from the training dataset
show5(train_loader)

# View image shape and label
data_iter = iter(train_loader)
images, labels = data_iter.next()

# Check the dimensions of one image and one label
print(f"Image shape: {images[0].shape}")  # Should be (1, 28, 28) for MNIST
print(f"Label: {labels[0].item()}")

```


    
![png](output_13_0.png)
    


    Image shape: torch.Size([1, 28, 28])
    Label: 9


## Build your Neural Network
Using the layers in `torch.nn` (which has been imported as `nn`) and the `torch.nn.functional` module (imported as `F`), construct a neural network based on the parameters of the dataset.
Use any architecture you like. 

*Note*: If you did not flatten your tensors in your transforms or as part of your preprocessing and you are using only `Linear` layers, make sure to use the `Flatten` layer in your network!


```python
def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),  # Convolutional layer with padding
        nn.ReLU(),                      # Activation function
        nn.AvgPool2d(2, stride=2),      # Pooling layer

        nn.Conv2d(6, 16, 5),            # Another convolutional layer
        nn.ReLU(),                      # Activation function
        nn.AvgPool2d(2, stride=2),      # Pooling layer
        
        nn.Flatten(),                   # Flatten the tensor for fully connected layers
        nn.Linear(400, 120),            # Fully connected layer
        nn.ReLU(),                      # Activation function
        nn.Linear(120, 84),             # Another fully connected layer
        nn.ReLU(),                      # Activation function
        nn.Linear(84, 10)               # Output layer with 10 classes (for MNIST digits)
    )
    return model
```

Specify a loss function and an optimizer, and instantiate the model.

If you use a less common loss function, please note why you chose that loss function in a comment.


```python
# Specify the loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = create_lenet().to(device)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(cnn.parameters(), lr=1e-3)  # Adam optimizer with learning rate 0.001
```

## Running your Neural Network
Use whatever method you like to train your neural network, and ensure you record the average loss at each epoch. 
Don't forget to use `torch.device()` and the `.to()` method for both your model and your data if you are using GPU!

If you want to print your loss **during** each epoch, you can use the `enumerate` function and print the loss after a set number of batches. 250 batches works well for most people!


```python
import matplotlib.pyplot as plt

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=5, print_every=250):
    model.train()  # Set the model to training mode
    train_losses = []  # To record training losses
    val_losses = []    # To record validation losses
    val_accuracies = [] # To record validation accuracies
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Training phase
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (i + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/print_every:.4f}")
                running_loss = 0.0
        
        # Store the average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase (optional)
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient calculation
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        model.train()  # Set the model back to training mode
    
    return train_losses, val_losses, val_accuracies

```

Plot the training loss (and validation loss/accuracy, if recorded).


```python
# Train the model and record losses
train_losses, val_losses, val_accuracies = train_model(cnn, train_loader, test_loader, criterion, optimizer, device, epochs=5)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

```

    Epoch [1/5], Batch [250/938], Loss: 0.7521
    Epoch [1/5], Batch [500/938], Loss: 0.2280
    Epoch [1/5], Batch [750/938], Loss: 0.1555
    Validation Loss: 0.0887, Accuracy: 97.04%
    Epoch [2/5], Batch [250/938], Loss: 0.0984
    Epoch [2/5], Batch [500/938], Loss: 0.0885
    Epoch [2/5], Batch [750/938], Loss: 0.0810
    Validation Loss: 0.0593, Accuracy: 98.00%
    Epoch [3/5], Batch [250/938], Loss: 0.0653
    Epoch [3/5], Batch [500/938], Loss: 0.0587
    Epoch [3/5], Batch [750/938], Loss: 0.0598
    Validation Loss: 0.0538, Accuracy: 98.20%
    Epoch [4/5], Batch [250/938], Loss: 0.0449
    Epoch [4/5], Batch [500/938], Loss: 0.0496
    Epoch [4/5], Batch [750/938], Loss: 0.0513
    Validation Loss: 0.0374, Accuracy: 98.70%
    Epoch [5/5], Batch [250/938], Loss: 0.0371
    Epoch [5/5], Batch [500/938], Loss: 0.0442
    Epoch [5/5], Batch [750/938], Loss: 0.0379
    Validation Loss: 0.0352, Accuracy: 98.79%



    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    


## Testing your model
Using the previously created `DataLoader` for the test set, compute the percentage of correct predictions using the highest probability prediction. 

If your accuracy is over 90%, great work, but see if you can push a bit further! 
If your accuracy is under 90%, you'll need to make improvements.
Go back and check your model architecture, loss function, and optimizer to make sure they're appropriate for an image classification task.


```python
def test_model(model, test_loader, device):
    model.eval() # Set the model to evaluation mode
    correct = 0 # To track the number of correct predictions
    total = 0 # To track the total number of samples
    
    with torch.no_grad(): # Disable gradient calculation during testing
        for images, labels in test_loader: # Loop through batches in test_loader
            images, labels = images.to(device), labels.to(device) # Move data to the device
            outputs = model(images) # Get predictions from the model
            
            # Get the predicted class (the one with the highest score)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0) # Count the total number of samples
            correct += (predicted == labels).sum().item() # Count correct predictions
    # Compute the accuracy as a percentage
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    
    return accuracy
```


```python
test_accuracy = test_model(cnn, test_loader, device)
```

    Accuracy on the test set: 98.79%


## Improving your model

Once your model is done training, try tweaking your hyperparameters and training again below to improve your accuracy on the test set!


```python
# Specify the loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = create_lenet().to(device)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(cnn.parameters(), lr=5e-4)  # Lower learning rate for better convergence

# Increase epochs to give the model more time to train, but not too high to avoid overfit. I increased to 10 and val loss starts to
# to increase, so I cut down to 6
train_losses, val_losses, val_accuracies = train_model(cnn, train_loader, test_loader, criterion, optimizer, device, epochs=6)


# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

```

    Epoch [1/6], Batch [250/938], Loss: 0.8533
    Epoch [1/6], Batch [500/938], Loss: 0.2727
    Epoch [1/6], Batch [750/938], Loss: 0.1922
    Validation Loss: 0.1250, Accuracy: 95.81%
    Epoch [2/6], Batch [250/938], Loss: 0.1289
    Epoch [2/6], Batch [500/938], Loss: 0.1156
    Epoch [2/6], Batch [750/938], Loss: 0.0972
    Validation Loss: 0.0770, Accuracy: 97.51%
    Epoch [3/6], Batch [250/938], Loss: 0.0834
    Epoch [3/6], Batch [500/938], Loss: 0.0732
    Epoch [3/6], Batch [750/938], Loss: 0.0799
    Validation Loss: 0.0649, Accuracy: 97.84%
    Epoch [4/6], Batch [250/938], Loss: 0.0649
    Epoch [4/6], Batch [500/938], Loss: 0.0644
    Epoch [4/6], Batch [750/938], Loss: 0.0575
    Validation Loss: 0.0514, Accuracy: 98.28%
    Epoch [5/6], Batch [250/938], Loss: 0.0567
    Epoch [5/6], Batch [500/938], Loss: 0.0471
    Epoch [5/6], Batch [750/938], Loss: 0.0480
    Validation Loss: 0.0389, Accuracy: 98.64%
    Epoch [6/6], Batch [250/938], Loss: 0.0396
    Epoch [6/6], Batch [500/938], Loss: 0.0425
    Epoch [6/6], Batch [750/938], Loss: 0.0428
    Validation Loss: 0.0439, Accuracy: 98.59%



    
![png](output_26_1.png)
    



    
![png](output_26_2.png)
    


## Saving your model
Using `torch.save`, save your model for future loading.


```python
# Define the path where I want to save the model
model_save_path = './mnist_model.pth'

# Save the model's state_dict
torch.save(cnn.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

```

    Model saved to ./mnist_model.pth

