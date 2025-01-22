## MNIST Neural Network

### Overview

This project implements a custom neural network to classify handwritten digits from the MNIST dataset. The MNIST dataset is a benchmark dataset for image classification tasks, containing 70,000 grayscale images of digits (28x28 pixels) labeled from 0 to 9. The project demonstrates the complete pipeline from data preprocessing to model training, validation, and evaluation. A Convolutional Neural Network (CNN) is designed and optimized to achieve high accuracy on this dataset.

### Features

- **Data Preprocessing**: Images are converted to tensors and normalized to improve model performance.
- **Custom CNN Architecture**: A simplified version of LeNet is implemented using PyTorch for multi-class classification.
- **Training and Validation**: Loss and accuracy metrics are tracked to monitor performance during training and validation phases.
- **Model Evaluation**: The trained model achieves high accuracy on the test dataset, close to state-of-the-art results.
- **Save and Reuse**: The trained model can be saved and reused for future inference tasks.
- **Hyperparameter Tuning**: Learning rate and epochs are optimized for better performance.

### Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/your-repo/mnist-neural-network.git
   cd mnist-neural-network
   ```

2. Install the required dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Ensure that your `PATH` includes the user installation directory:

   ```python
   import os
   os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
   ```

   **Important**: Restart the Python kernel after this step.

### Usage

1. **Data Loading**: The MNIST dataset will be automatically downloaded and preprocessed using PyTorch's `torchvision` module.

2. **Model Training**:
   Run the training script to train the neural network on the MNIST dataset:
   ```python
   python train.py
   ```

3. **Evaluate Model**:
   Evaluate the model's accuracy on the test dataset:
   ```python
   python evaluate.py
   ```

4. **Save Model**:
   Save the trained model for future use:
   ```python
   python save_model.py
   ```

5. **Load Model**:
   Load a saved model for inference:
   ```python
   python load_model.py
   ```

### Results

- **Training Accuracy**: ~98.79%
- **Validation Accuracy**: ~98.59%
- **Test Accuracy**: ~98.79%

These results demonstrate that the model performs well and achieves competitive accuracy on the MNIST dataset.

### Model Architecture

The neural network is a simplified version of the LeNet architecture:

1. **Convolutional Layers**: Two layers with ReLU activation and average pooling.
2. **Fully Connected Layers**: Three layers with ReLU activations.
3. **Output Layer**: Softmax activation for multi-class classification (10 classes).

### Directory Structure

```plaintext
├── data/                 # Directory for MNIST dataset
├── train.py              # Script for training the model
├── evaluate.py           # Script for evaluating the model
├── save_model.py         # Script for saving the model
├── load_model.py         # Script for loading the model
├── mnist_model.pth       # Saved model file
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
```

### Dependencies

The project requires the following Python libraries:

- PyTorch
- torchvision
- NumPy
- Matplotlib
- tqdm

Install all dependencies using the `requirements.txt` file.

### Future Improvements

- Experiment with advanced architectures like ResNet or EfficientNet.
- Implement data augmentation techniques to further improve model generalization.
- Extend the project to other datasets or tasks such as CIFAR-10 or CIFAR-100.
- Optimize training for deployment on resource-constrained devices.

