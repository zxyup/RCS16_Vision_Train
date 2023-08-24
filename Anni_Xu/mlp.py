import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        )

    def forward(self,x):
        x=x.view(x.size(0),-1)
        out = self.layers(x)
        return out
    
input_size=784
hidden_size=128
output_size=10
learning_rate=0.001
batch_size=64
num_epochs=10

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transform,download=True)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

model=MLP(input_size,hidden_size,output_size)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

mlp_loss=[]
mlp_train=[]
mlp_test=[]
for epoch in range(num_epochs):
    
    sumloss=0
    for i,(images,labels) in enumerate(train_loader):
        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sumloss+=loss.item()       
        #if(i+1)%100==0:
            #print(f"Epoch[{epoch+1}/{num_epochs}],Step[{i+1}/{len(train_loader)}],Loss:{loss.item():.4f}")
    mlp_loss.append(sumloss)


    correct1=0
    total1=0
    with torch.no_grad():
        for images,labels in train_loader:
            outputs1=model(images)
            _1,predicted1=torch.max(outputs1.data,1)
            total1+=labels.size(0)
            correct1+=(predicted1==labels).sum().item()
    print(f"Accuracy of the model on the 10000 train images:{100*correct1/total1:.2f}%")
    mlp_train.append(correct1/total1)

    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print(f"Accuracy of the model on the 10000 test images:{100*correct/total:.2f}%")
    mlp_test.append(correct/total)

plt.subplot(131)
plt.plot([i + 1 for i in range(num_epochs)], mlp_train)
plt.title('train')
plt.subplot(132)
plt.plot([i + 1 for i in range(num_epochs)], mlp_test)
plt.title('test')
plt.subplot(133)
plt.plot([i + 1 for i in range(num_epochs)], mlp_loss)
plt.title('loss')

plt.tight_layout()
plt.show()