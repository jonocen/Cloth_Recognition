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

# Define a simple Network
class NETWORK(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.STACK = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, (4, 4)),  
            torch.nn.MaxPool2d((4, 4)),
            torch.nn.Conv2d(4, 4, (2, 2),),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64 ,10)
        )

    def forward(self, x):
        return self.STACK(x)

# Instantiate and move model to the appropriate device
model = NETWORK().to(device=device)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

DataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

opti = torch.optim.Adam(model.parameters())
lossfn = torch.nn.CrossEntropyLoss()

def TRAINING(model:torch.nn.Module, epoches):
    count = 0
    model.train(True)
    for I in range(epoches):
        for x, y in DataLoader:
            x, y = x.to(device), y.to(device)
            pred =  model(x) #we let the model go throug x
            loss = lossfn(pred, y) #we calculate the loss
            loss.backward() #we backward the loss
            opti.step()#the optimizer takes a step
            opti.zero_grad()#the optimizer zero grands
            print(f"Epoch [{I + 1}/{epoches}], Step [{count}], Loss: {loss.item()}")
TRAINING(model, 10)
torch.save(model.state_dict(), "MODEL.pth")
