import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


#print("训练集的长度:{}".format(len(train_data)))
#print("测试集的长度:{}".format(len(test_data)))

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
model = Model().cuda()

#添加tensorboard可视化数据
#writer = SummaryWriter('../logs_tensorboard')

# 损失函数
loss = nn.CrossEntropyLoss().cuda()

# 优化器
optimizer  = torch.optim.SGD(model.parameters(),lr=0.025,)

i = 1 # 用于绘制测试集的tensorboard
mlp_loss=[]
mlp_train=[]
mlp_test=[]
num_epochs=30

# 开始循环训练
for epoch in range(num_epochs):

    if epoch >20:  # 更新一次学习率        
        for params in optimizer.param_groups:             
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9            
            params['lr'] =   0.01          
            # params['weight_decay'] = 0.5  # 当然也可以修改其他属性

    num_time = 0 # 记录看看每轮有多少次训练
    print('开始第{}轮训练'.format(epoch+1))
    model.train() # 也可以不写，规范的话是写，用来表明训练步骤
    sumloss = 0 # 记录总体损失值
    for data in train_dataloader:
        # 数据分开 一个是图片数据，一个是真实值
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        # 拿到预测值
        output = model(imgs)
        # 计算损失值
        loss_in = loss(output,targets)
        # 优化开始~ ~ 先梯度清零
        optimizer.zero_grad()
        # 反向传播+更新
        loss_in.backward()
        optimizer.step()
        #num_time +=1
        sumloss+=loss_in.item()   
        #if num_time % 100 == 0:
            #writer.add_scalar('看一下训练集损失值',loss_in.item(),num_time)
    mlp_loss.append(sumloss)
    

    accurate1 = 0
    #model.eval() # 也可以不写，规范的话就写，用来表明是测试步骤
    with torch.no_grad():
        for data in train_dataloader:
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
            imgs , targets = data
            imgs = imgs.cuda()
            targets1 = targets.cuda()
            output1 = model(imgs)
            loss_in1 = loss(output1,targets1)

            #sumloss += loss_in
            #print('这里是output',output)
            accurate1 += (output1.argmax(1) == targets1).sum().item()
    print('第{}轮测试集的正确率:{:.2f}%'.format(epoch+1,accurate1/len(train_data)*100))
    mlp_train.append(accurate1/len(train_data))

        # 每轮训练完成跑一下测试数据看看情况
    accurate = 0
    model.eval() # 也可以不写，规范的话就写，用来表明是测试步骤
    with torch.no_grad():
        for data in test_dataloader:
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
            imgs , targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs)
            loss_in = loss(output,targets)

            #sumloss += loss_in
            #print('这里是output',output)
            accurate += (output.argmax(1) == targets).sum().item()

    print('第{}轮测试集的正确率:{:.2f}%'.format(epoch+1,accurate/len(test_data)*100))
    mlp_test.append(accurate/len(test_data))
        #writer.add_scalar('看一下测试集损失',sum_loss,i)
        #writer.add_scalar('看一下当前测试集正确率',accurate/len(test_data)*100,i)
    i +=1

        #torch.save(model,'../model_pytorch/model_{}.pth'.format(epoch+1))
    print("第{}轮模型训练数据已保存".format(epoch+1))

#writer.close()
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