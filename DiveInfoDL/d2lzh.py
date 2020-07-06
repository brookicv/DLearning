import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import sys

# ImageAssistant


def load_data_fashion_mnist(batch_size,resize=None):

    trans = []
    if resize:
        trans.append((transforms.Resize(size=resize)))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST("../../Datasets/FashionMNIST",train=True,download=True,transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST("../../Datasets/FashionMNIST",train=False,download=True,transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True,num_workers=4)

    return train_iter,test_iter

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankel boot']

    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images,labels):

    _,figs = plt.subplots(1,10,figsize=(12,12))
    for f,img,lbl in zip(figs,images,labels):
        img = img.view((28,28)).numpy()
        f.imshow(img)
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

    plt.show()



def evaluate_accuracy(data_iter, net):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval() # 评估模式，关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
        else:
            if ('is_training' in net.__code__.co_varnames):  # 有is_training这个参数
                acc_sum += (net(X,is_training=False).argmax(dim=1) ==y).float().sum().item()

        n += y.shape[0]

    return acc_sum / n


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):

    print("changeing train")
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for X, y in train_iter:

            y_hat = net(X)
            l = loss(y_hat,y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            test_acc = evaluate_accuracy(test_iter,net)

        print("epoch %d,loss %.4f,train acc %.3f,test_acc %.3f" % (epoch +1,train_l_sum / n ,train_acc_sum / n,test_acc))

def train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net = net.to(device)
    print("training on ",device)

    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start = 0.0,0.0,0,time.time()

        for x,y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()

            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter,net)

        print('epoch %d, loss %.4f, train acc %.3f,test acc %.3f,time %.1f' %(epoch + 1,train_l_sum / batch_count,train_acc_sum / n ,test_acc,time.time()-start))

def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,legend=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=':')
        plt.legend(legend)


def corr2d(x, k):
    h, w = k.shape

    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()

    return y


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可以将池化窗口设置为输入的宽和高实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])
