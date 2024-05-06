# Importing necessary libraries
from data import X_train, X_test, y_train, y_test
from data import DataCreator, device
from model import CircleClassifier, accuracy
from torch import nn
import torch

# Create an insatnce of CircleClassifier
circleClassifier = CircleClassifier(input_features=2, output_features=1, hidden_units=8).to(device)

# Setup the loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = circleClassifier.parameters(), lr=0.3)

# Set the number of epochs
epochs = 1000

# Training and testing loop
for epoch in range(epochs):
    ### Train loop
    circleClassifier.train()

    # 1. Forward pass
    y_logits = circleClassifier(X_train).squeeze(dim=1) # Logits = raw data
    y_preds = torch.round(torch.sigmoid(y_logits)) # Turn raw data into predictions

    # 2. Calculate the loss / acc
    loss = loss_fn(y_logits, y_train) 
    acc = accuracy(y_true=y_train, y_preds=y_preds)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Test loop
    circleClassifier.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = circleClassifier(X_test).squeeze(dim=1)
        test_preds = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate loss / acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy(y_test, test_preds) 
        
    # Print everything
    if epoch % 100 == 0:
        print(f"Epoch : {epoch} | Loss : {loss:.5f} | Acc : {acc:.2f}% | Test Loss : {test_loss:.5f} | Test Acc : {test_acc:.2f}%")

# Make predictions
circleClassifier.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(circleClassifier(X_test))).squeeze(dim=1)

# Compare predictions and true data
print(y_preds[:10])
print(y_test[:10])

# Create an instance of DataCreater to visualize the model
dataCreator = DataCreator()
dataCreator.visualize_data(circleClassifier, X_train, X_test, y_train, y_test)