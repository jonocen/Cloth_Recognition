import torch
import numpy as np
import torchvision


# Check if CUDA is available
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")
else:
    device = "cpu"
    print("Using CPU")

# Define the transformation
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Load the FashionMNIST dataset
dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Define the neural network
class NETWORK(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.STACK = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, (4, 4)),  
            torch.nn.MaxPool2d((4, 4)),
            torch.nn.Conv2d(4, 4, (2, 2)),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.STACK(x)

# Instantiate the model and load the weights
model = NETWORK().to(device=device)
model.load_state_dict(torch.load('MODEL.pth', weights_only=False))

# Get a sample from the dataset
x, y = dataset[3]
x = x.to(device) 

# Set the model to evaluation mode
model.eval()

# Make a prediction
output = model(x.unsqueeze(0))  

# Get the predicted class
predicted_class = torch.argmax(output, dim=1).item() 

# Print the results
print(f"Model output: {output}, Predicted class: {predicted_class}")
print(f"True class {y}")