import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import sys
import random
import zipfile

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

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1):
        super(Residual,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            x = self.conv3(x)

        return F.relu(y + x)
################################################################
## chapter 6 language model
################################################################


# 周杰伦歌词数据集
def load_data_jay_lyrics():
    with zipfile.ZipFile("../data/jaychou_lyrics.txt.zip")as zin:
        with zin.open("jaychou_lyrics.txt") as f:
            corpus_chars = f.read().decode("utf-8")


    # 这个数据集有６万多个字符，为了打印方便，将换行符换成空格
    corpus_chars = corpus_chars.replace("\n", " ").replace("\r", " ")
    corpus_chars = corpus_chars[0:20000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]

    return corpus_indices,char_to_idx,idx_to_char,vocab_size


def data_iter_random(corpus_indices,batch_size,num_steps,device=None):
    # -1是因为输出的索引ｘ市相应的输入的索引ｙ+1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size

    examples_indices = list(range(num_examples))
    random.shuffle(examples_indices)

    def _data(pos):
        return corpus_indices[pos:pos + num_steps]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = examples_indices[i: i + batch_size]
        x = [_data(j * num_steps) for j in batch_indices]
        y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device)

def data_iter_consecutive(corpus_indices,batch_size,num_stpes,device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus_indices = torch.tensor(corpus_indices,dtype=torch.float32,device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[0:batch_size * batch_len].view(batch_size,batch_len)

    epoch_size = (batch_len - 1) // num_stpes

    for i in range(epoch_size):
        i = i * num_stpes
        x = indices[:,i : i + num_stpes]
        y = indices[:,i + 1 : i + num_stpes + 1]
        yield x,y


def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(x, n_class):
    # x shape:(batch,seq_len),
    # output: seq_len elements of (batch,n_class)

    return [one_hot(x[:, i], n_class) for i in range(x.shape[1])]


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens,
                vocab_size, device, idx_to_char, char_to_idx):

    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一个时间步的输出作为下一个时间步的输入
        x = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        (y, state) = rnn(x, state, params)  #计算和更新隐藏状态
        #下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(y[0].argmax(dim=1).item()))

    return "".join([idx_to_char[i] for i in output])

def grad_clipping(params,theta,device):
    norm = torch.tensor([0.0],device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive

    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 使用相邻采样，在epoch开始时初始化隐藏层
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for x, y in data_iter:
            #　随机采样，在每个小批量更新钱初始化隐藏状态
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                for s in state:
                    s.detach_()
            inputs = to_onehot(x, vocab_size)
            (outputs, state) = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim=0)
            y = torch.transpose(x, 0, 1).contiguous().view(-1)
            l = loss(outputs, y.long())

            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)
            d2l.sgd(params, lr, 1)

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print("epoch %d,perplexity %f,time %.2f sec" %
                  (epoch + 1, math.exp(l_sum / n), time.time() - start))

            for prefix in prefixes:
                print(" -",predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                num_hiddens, vocab_size, device, idx_to_char,
                                char_to_idx))
