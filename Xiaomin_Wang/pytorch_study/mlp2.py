# MLP实现手写字体识别
# 效果图片按训练集准确率，测试集准确率，训练集损失排一排
# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义一个多层感知器（MLP）类，继承自 nn.Module
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        # 构建网络结构
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # 定义前向传播
        x = x.view(x.size(0), -1) # Flatten the input tensor
        out = self.layers(x)
        return out

# 超参数设置
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型、损失函数和优化器对象
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

y_sum_loss = []
train_correct = []
test_correct = []
# 开始训练和测试
for epoch in range(num_epochs):
    sum_loss = 0
    correct_ = 0
    total_ = 0
    # 进行训练 
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        sum_loss += loss 
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算训练准确率
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_ += labels.size(0)
        correct_ += (predicted == labels).sum().item()
        # 输出损失信息
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    # 计算每一轮训练的准确率并添加到数组中
    train_correct.append(100*correct_/total_)

    # 计算每一轮训练的损失率并添加到数组中
    y_sum_loss.append(sum_loss.detach().numpy())

    # 测试模型
    correct = 0
    total = 0
    model.eval()
    # 表明当前的计算不需要反向传播
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_correct.append(100*correct/total)

#开始绘制三个图像
plt.figure(figsize=(12,4))
x = [i + 1 for i in range(num_epochs)]
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
print(f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%")

