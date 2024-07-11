import torch
from torch import nn
import matplotlib.pyplot as plt

def plot_mygraph(*args):
    fig = plt.figure(figsize = (6,6), facecolor = "lightskyblue")
    fig.suptitle('My Graph')
    ax = fig.add_subplot(1,1,1)
    plt.scatter(*args)
    plt.show()

weight = 0.7
bias = 0.3
torch.manual_seed(42)
x_feautres = torch.randn(10,1)
y_labels = (weight * x_feautres) + bias
x_train = x_feautres[:7]
y_train = y_labels[:7]
x_test = x_feautres[7:]
y_test = y_labels[7:]

class Neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=1, out_features=4),
            nn.Linear(in_features=4,out_features=1)
                                    )
    def forward(self, x):
        return self.layer1(x)    
    
model1 = Neural_network()

y_eval = model1(x_feautres)
print(y_eval, y_labels)


lossfn = nn.L1Loss()
optimizer = torch.optim.SGD(model1.parameters(),lr = 0.01)

model1.train()

epochs = 200

for i in range(epochs):
    model1.train()

    y_predictions = model1(x_train)

    loss = lossfn(y_predictions, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

with torch.no_grad():
    model1.eval()

    y_eval = model1(x_test)

print(y_eval, y_test)    
    
