import torch 
import torch.nn as nn

class LeNet_half(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet_half, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, 5, stride=1, padding=0)
        self.conv1.bias.data.normal_(0, 0.1)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = nn.ReLU()   
        
        self.pad = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(3, 8, 5, stride=1, padding=0)
        self.conv2.bias.data.normal_(0, 0.1)
        self.conv2.bias.data.fill_(0)  
        
        self.fc1 = nn.Linear(200,120)
        self.fc1.bias.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0) 
        
        self.fc2 = nn.Linear(120,84)
        self.fc2.bias.data.normal_(0, 0.1)
        self.fc2.bias.data.fill_(0) 
        
        self.fc3 = nn.Linear(84,num_classes)
        self.fc3.bias.data.normal_(0, 0.1)
        self.fc3.bias.data.fill_(0) 
        
        self.soft = nn.Softmax()
        
    def forward(self, x):
        layer1 = self.pad(self.relu(self.conv1(x)))
        layer2 = self.pad(self.relu(self.conv2(layer1)))

        flatten = layer2.view(-1, 8*5*5)
        fully1 = self.relu(self.fc1(flatten))
        
        fully2 = self.relu(self.fc2(fully1))
        
        logits = self.fc3(fully2)
        #softmax_val = self.soft(logits)

        return logits

model = LeNet_half(num_classes=10)

