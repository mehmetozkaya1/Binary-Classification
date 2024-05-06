# Importing necessary libraries
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import numpy as np

"""

# make_circles method is used to create a binary classification dataset
# train_test_split method splits the data into train and test data
# matplotlib library is used to visualize the data

"""
# Number of examples
n_samples = 1000

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a DataCreator class to create the dataset
class DataCreator:
    def __init__(self, n_samples : int = 1000):
        """
        
        Args : 
        n_samples (int) : Number of examples in the dataset
        
        """
        self.n_samples = n_samples # Set the number of examples

    # The method that creates the dataset
    def create_dataset(self):
        X, y = make_circles(n_samples=self.n_samples, noise=0.03)

        print(f"Number of examples : {len(X)}")
        print(f"Number of labels : {len(y)}")

        return X, y
    
    # The method that splits the dataset into train and test data
    def split_dataset(self, X : np.array, y : np.array, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        print(f"Train data length = {len(X_train)}")
        print(f"Test data length = {len(X_test)}")

        return X_train, X_test, y_train, y_test
    
    # The method that visualize the data in a scatter graph
    def visualize_data_scatter(self, X : np.array, y : np.array):
        plt.scatter(x = X[:, 0], y = X[:, 1], c=y, cmap=plt.cm.RdYlBu)
        plt.show()

    # The method that turns the data into PyTorch Tensors. Also it sets the data to device
    def data_to_tensors(self, x : list):
        x = torch.from_numpy(x).type(torch.float)
        x = x.to(device=device)
        return x 
    
    # The method that plots the decision boundary
    def plot_decision_boundary(self, model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
        # Put everything to CPU (works better with NumPy + Matplotlib)
        model.to("cpu")
        X, y = X.to("cpu"), y.to("cpu")

        # Setup prediction boundaries and grid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

        # Make features
        X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

        # Make predictions
        model.eval()
        with torch.inference_mode():
            y_logits = model(X_to_pred_on)

        # Test for multi-class or binary and adjust logits to prediction labels
        if len(torch.unique(y)) > 2:
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
        else:
            y_pred = torch.round(torch.sigmoid(y_logits))  # binary

        # Reshape preds and plot
        y_pred = y_pred.reshape(xx.shape).detach().numpy()
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

    def visualize_data(self, model, x_train, x_test, y_train, y_test):
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.title("Train")
        self.plot_decision_boundary(model, x_train, y_train)
        plt.subplot(1,2,2)
        plt.title("Test")
        self.plot_decision_boundary(model, x_test, y_test)
        plt.show()

# Create the DataCreator object
dataCreator = DataCreator(n_samples=1000)

# Create the dataset
X, y = dataCreator.create_dataset()

# Visualize the data
dataCreator.visualize_data_scatter(X, y)

# Split the data into training and testing data
X_train, X_test, y_train, y_test = dataCreator.split_dataset(X, y, 0.2)

# Convert the data into PyTorch Tensors
X_train = dataCreator.data_to_tensors(X_train)
X_test = dataCreator.data_to_tensors(X_test)
y_train = dataCreator.data_to_tensors(y_train)
y_test = dataCreator.data_to_tensors(y_test)   