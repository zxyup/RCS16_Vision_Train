import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
np.random.seed(50)
x=2 *np.random.rand(100,1)
y=1+5*x+np.random.randn(100,1)*0.3
x=torch.from_numpy(x).float()
y=torch.from_numpy(y).float()
dataset=TensorDataset(x,y)

dataloader=DataLoader(dataset,batch_size=25,shuffle=True)
print('Len of DataLoader',len(dataloader))
for index,(data,label) in enumerate(dataloader):
    print(f'index={index},num={len(data):2}')#,data={data},label={label}
epoch=400
lr=0.0054
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
print(w)
print(b)
Loss=[]
for epoch in range(1,epoch+1):
    sum_loss=0
    for batch_id,(bx,by) in enumerate(dataloader):
        h =bx*w + b
        loss= torch.mean((h-by)**2)
        # print('Loss',loss)
        sum_loss+=loss.item()
        loss.backward()
        # print('wg:',w.grad.data,'bg:',b.grad.data)
        # print('w:',w.data,'b:',b.data)
        w.data-=lr*w.grad.data
        b.data-=lr*b.grad.data
        # print('w:',w.data,'b:',b.data)
        w.grad.zero_()
        b.grad.zero_()
        # print('wg:',w.grad.data,'bg:',b.grad.data)
        # print(f'epoch:{epoch},batch:{batch_id},loss={loss}')
    # print(f'epoch:{epoch},loss={sum_loss}')
    Loss.append(sum_loss)
print(loss.item())
print(f'w:{w.item()}')
print(f'b:{b.item()}')
Loss_x=[i for i in range(1,epoch+1)]
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("Xing Xia")
plt.plot(Loss_x,Loss)
w=w.item()
b=b.item()
xx=np.linspace(0,2,100)
h=w * xx+ b

plt.subplot(1,2,2)
plt.title("Xing Xia")
plt.plot(xx,h)
plt.scatter(x,y,marker='+',color='red')
plt.show()