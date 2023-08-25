import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='D:/Data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='D:/Data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


print("训练集的长度:{}".format(len(train_data)))
print("测试集的长度:{}".format(len(test_data)))

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
writer = SummaryWriter('../logs_tensorboard')

# 损失函数
loss = nn.CrossEntropyLoss().cuda()

# 优化器
optimizer  = torch.optim.SGD(model.parameters(),lr=0.01,)

i = 1 # 用于绘制测试集的tensorboard
EPOCH = 20
y_sum_loss = []
train_correct = []
test_correct = []
# 开始循环训练
for epoch in range(EPOCH):
    num_time = 0 # 记录看看每轮有多少次训练
    sum_loss = 0 # 记录总体损失值
    accurate_ = 0
    print('开始第{}轮训练'.format(epoch+1))
    # 进行训练
    model.train() # 也可以不写，规范的话是写，用来表明训练步骤
    for data in train_dataloader:
        # 数据分开 一个是图片数据，一个是真实值
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        # 拿到预测值
        output = model(imgs)
        # 计算损失值
        loss_in = loss(output,targets)
        sum_loss += loss_in
        # 优化开始~ ~ 先梯度清零
        optimizer.zero_grad()
        # 反向传播+更新
        loss_in.backward()
        optimizer.step()
        num_time +=1
        if num_time % 100 == 0:
            writer.add_scalar('看一下训练集损失值',loss_in.item(),num_time)
        accurate_ += (output.argmax(1) == targets).sum()
    accurate_ = accurate_.cpu()
    train_correct.append(accurate_.detach().numpy()/len(train_data)*100)
    sum_loss = sum_loss.cpu()
    y_sum_loss.append(sum_loss.detach().numpy())
    # 每轮训练完成跑一下测试数据看看情况
    accurate = 0
    # 进行测试
    model.eval() # 也可以不写，规范的话就写，用来表明是测试步骤
    with torch.no_grad():
        for data in test_dataloader:
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
            imgs , targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs)
            loss_in = loss(output,targets)
            # 计算测试准确率
            accurate += (output.argmax(1) == targets).sum()
    accurate = accurate.cpu()
    # 往测试准确率的数组中添加数据
    test_correct.append(accurate.detach().numpy()/len(test_data)*100)
    print('第{}轮测试集的正确率:{:.2f}%'.format(epoch+1,accurate/len(test_data)*100))

    writer.add_scalar('看一下测试集损失',sum_loss,i)
    writer.add_scalar('看一下当前测试集正确率',accurate/len(test_data)*100,i)
    i += 1

    # torch.save(model,'../model_pytorch/model_{}.pth'.format(epoch+1))
    # print("第{}轮模型训练数据已保存".format(epoch+1))

#开始绘制三个图像
plt.figure(figsize=(12,3 ))
x = [i + 1 for i in range(EPOCH)]
#损失图像
plt.subplot(133)
plt.plot(x, y_sum_loss,color='red', label="train loss")
plt.title('train_loss')
plt.legend()
# 训练准确率
plt.subplot(131)
plt.plot(x, train_correct, label="train correct")
plt.title('train correct')
plt.legend()
# 测试准确率
plt.subplot(132)
plt.plot(x, test_correct, label="test correct")
plt.title('test correct')
plt.legend()
plt.tight_layout()
plt.show()

writer.close()
