# RCS16_Vision_Train


## 队员：金海洋

### 第一节 opencv实践
效果图

！[扇叶捕捉](../picture/left4.png)

##### 代码部分
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2 as cv
    #加载图像
    img1 = cv.imread('Target.png',cv.IMREAD_GRAYSCALE)          # 索引图像
    img2 = cv.imread('mmm.png',cv.IMREAD_GRAYSCALE) # 训练图像
    ##### 初始化SIFT描述符
    sift = cv.xfeatures2d.SIFT_create()
    ##### 基于SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    ##### 默认参数初始化BF匹配器
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    ##### 应用比例测试
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    ##### cv.drawMatchesKnn将列表作为匹配项。
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
*** 第二节 多次函数回归 ***
### 效果图

！[线性回归](../picture/buhuo.png)


## 代码部分
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 输入图像是单通道，conv1 kenrnel size=5*5，输出通道 6
            self.conv1 = nn.Conv2d(1, 6, 5)
            # conv2 kernel size=5*5, 输出通道 16
            self.conv2 = nn.Conv2d(6, 16, 5)
            # 全连接层
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # max-pooling 采用一个 (2,2) 的滑动窗口
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            # 除了 batch 维度外的所有维度
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    net = Net()
    print(net)
    ###### 清空所有参数的梯度缓存，然后计算随机梯度进行反向传播
    net.zero_grad()
    #数据准备
    np.random.seed(42)
    x=2 *np.random.rand(100,1)
    x-=1
    y=1+5*x*x*x+np.random.randn(100,1)*0.1
    plt.scatter(x,y,marker='+',color='blue')
    #数据处理
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    x=torch.from_numpy(x).float()
    y=torch.from_numpy(y).float()
    dataset=TensorDataset(x,y)

    dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
    print('Len of DataLoader',len(dataloader))
    for index,(data,label) in enumerate(dataloader):
        print(f'index={index},num={len(data):2}')#,data={data},label={label}
    epoch=761

    lr=0.0009999

    w=torch.randn(1,requires_grad=True)
    b=torch.randn(1,requires_grad=True)
    c=torch.randn(1,requires_grad=True)
    d=torch.randn(1,requires_grad=True)
    print(w)
    print(b)
    print(c)
    print(d)
    Loss=[]

    for epoch in range(1,epoch+1):
        sum_loss=0
        for batch_id,(bx,by) in enumerate(dataloader):
            h =w*(bx**3) + b*(bx**2)+bx*c+d
            loss= torch.mean((h-by)**2)
            sum_loss+=loss.item()
            loss.backward()
            w.data-=lr*w.grad.data
            b.data-=lr*b.grad.data
            c.data-=lr*b.grad.data
            d.data-=lr*b.grad.data
            w.grad.zero_()
            b.grad.zero_()
            c.grad.zero_()
            d.grad.zero_()
        Loss.append(sum_loss)

    Loss_x=[i for i in range(1,epoch+1)]
    plt.subplot(121)
    plt.plot(Loss_x,Loss)
    plt.title("JinHaiYang")
    #plt.show()
    w=w.item()
    b=b.item()
    c=c.item()
    d=d.item()
    xx=np.linspace(-1,1,100)
    h =w*(xx**3) + b*(xx**2)+xx*c+d
    plt.subplot(122)
    plt.plot(xx,h)
    plt.scatter(x,y,marker='+',color='red')
    plt.title("JinHaiYang")
    plt.show()

*** 第三节 多层感知器（MLP） ***

### 效果图

！[MLP](../picture/MLP.PNG)

## 代码部分

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 输入图像是单通道，conv1 kenrnel size=5*5，输出通道 6
            self.conv1 = nn.Conv2d(1, 6, 5)
            # conv2 kernel size=5*5, 输出通道 16
            self.conv2 = nn.Conv2d(6, 16, 5)
            # 全连接层
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # max-pooling 采用一个 (2,2) 的滑动窗口
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            # 除了 batch 维度外的所有维度
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    net = Net()
    print(net)
    ###### 清空所有参数的梯度缓存，然后计算随机梯度进行反向传播
    net.zero_grad()
    #数据准备
    np.random.seed(42)
    x=2 *np.random.rand(100,1)
    x-=1
    y=1+5*x*x*x+np.random.randn(100,1)*0.1
    plt.scatter(x,y,marker='+',color='blue')
    #数据处理
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    x=torch.from_numpy(x).float()
    y=torch.from_numpy(y).float()
    dataset=TensorDataset(x,y)

    dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
    print('Len of DataLoader',len(dataloader))
    for index,(data,label) in enumerate(dataloader):
        print(f'index={index},num={len(data):2}')#,data={data},label={label}

    def PlotMLP(net,  input_x, x):
        
        net_eval = net.eval()
        
        net_y = net_eval(input_x)

        net_y_np=net_y.cpu().detach().numpy()

    epoch=760
    lr=0.0009999
    w=torch.randn(1,requires_grad=True)
    b=torch.randn(1,requires_grad=True)
    c=torch.randn(1,requires_grad=True)
    d=torch.randn(1,requires_grad=True)
    print(w)
    print(b)
    print(c)
    print(d)
    Loss=[]
    for epoch in range(1,epoch+1):
        sum_loss=0
        for batch_id,(bx,by) in enumerate(dataloader):
            h =w*(bx**3) + b*(bx**2)+bx*c+d
            loss= torch.mean((h-by)**2)
            # print('Loss',loss)
            sum_loss+=loss.item()
            loss.backward()
            # print('wg:',w.grad.data,'bg:',b.grad.data)
            # print('w:',w.data,'b:',b.data)
            w.data-=lr*w.grad.data
            b.data-=lr*b.grad.data
            c.data-=lr*b.grad.data
            d.data-=lr*b.grad.data
            # print('w:',w.data,'b:',b.data)
            w.grad.zero_()
            b.grad.zero_()
            c.grad.zero_()
            d.grad.zero_()
            # print('wg:',w.grad.data,'bg:',b.grad.data)
            # print(f'epoch:{epoch},batch:{batch_id},loss={loss}')
        # print(f'epoch:{epoch},loss={sum_loss}')
        Loss.append(sum_loss)
    # print(loss.item())
    # print(f'w:{w.item()}')
    # print(f'b:{b.item()}')
    Loss_x=[i for i in range(1,epoch+1)]
    plt.subplot(221)
    plt.plot(Loss_x,Loss)
    plt.title("JinHaiYang")
    #plt.show()
    w=w.item()
    b=b.item()
    c=c.item()
    d=d.item()
    xx=np.linspace(-1,1,100)
    h =w*(xx**3) + b*(xx**2)+xx*c+d


    plt.subplot(222)
    plt.plot(xx,h)
    plt.scatter(x,y,marker='+',color='red')
    plt.title("JinHaiYang")
    #plt.show()
    plt.subplot(223)
    plt.title("evaluation")
    plt.plot(x, torch.sin(x), label="original", linewidth=3)
    plt.plot(x, [y[0] for y in net_y_np], label='MLP',color='red')
    plt.legend()
    plt.tight_layout()
    plt.show()
    PlotMLP(net,input_x,x)