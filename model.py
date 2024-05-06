# Importing necessary libraries
import torch
from torch import nn

# The class that operates binary classification
class CircleClassifier(nn.Module):
    def __init__(self, input_features : int, output_features : int, hidden_units : int = 8):

        """
        
        Args : 

        input_features (int) : Number of input features
        output_features (int) : Number of output features
        hidden_units (int) : Number of hidden units in each layer
        
        """

        super().__init__()
        self.layer_stack = nn.Sequential( # Neural Network structure
            nn.Linear(in_features = input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = output_features)
        )

    def forward(self, X : torch.Tensor): # Forward method
        return self.layer_stack(X)
    
# A function that computes the accuracy of the model
def accuracy(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct / len(y_preds)) * 100
    return acc