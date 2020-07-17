
import numpy as np
import torch.nn as nn
import torch
import cv2


def generate_anchors(base_size=16,ratios=[0.5,1,2],scales=2**np.arange(3,6)):
    """
    base_size: grid cell的大小，从原图像下采样得到。
    生成的anchor box是以原图像的尺寸为基准的
    """
    base_anchor = np.array([1,1,base_size,base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor,ratios) # 生成不同ratio的anchor box
    # 组合不同的scale
    anchors = np.vstack([_scale_enum(ratio_anchors[i,:],scales) for 
                        i in range(ratio_anchors.shape[0])])

    return anchors
    
def _whctrs(anchor):
    """
    将anchor表示为(x,y,w,h).
    (x,y)为中心点，(w,h)为宽和高

    Input: 
        anchor，边框左上角和右下角的坐标[x1,y1,x2,y2]
    Return:
        (x,y,w,h)
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    return (x_ctr,y_ctr,w,h)

def _mkanchors(ws,hs,x_ctr,y_ctr):
    """
    给定anchro的中心位置(x_ctr,y_ctr)以及高度，宽度的列表ws，hs，
    生成对应的anchor列表。anchor的表示方式为左上角和右下角的坐标
    """
    ws = ws[:,np.newaxis]
    hs = hs[:,np.newaxis]

    anchors = np.hstack((x_ctr - 0.5*(ws-1),
                        y_ctr - 0.5*(hs-1),
                        x_ctr + 0.5 *(ws-1),
                        y_ctr + 0.5 *(hs-1)))

    return anchors

def _ratio_enum(anchor,ratios):
    """
    生成不同纵横比的anchor
    Input:
        anchor: base anchor,宽和高相等。在该anchor的
                基础上，根据ratio生成不同的anchor box。

                w = sqrt(a/ratio),h = w * ratio = sqrt(a x ratio)
    """
    x_ctr,y_ctr,w,h = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    # 求出不同ratio的宽和高
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws,hs,x_ctr,y_ctr)

    return anchors

def _scale_enum(anchor,scales):
    """
    生成不同scale的anchor
    Input:
        scales，有两种定义方式，一种是针对宽和高的倍数；
                另一种，是anchor box的面积。 例如scale = 16,宽和高都放大4倍
    """
    x_ctr,y_ctr,w,h = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

def parse_cfg(cfgfile):

    file = open(cfgfile, "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0] # 移除空行
    lines = [x for x in lines if x[0] != "#"] # 移除注释行
    lines = [x.rstrip().lstrip() for x in lines]  #　移除两头的空格
    
    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # 新的block的开始符号，就是上一个block结束
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:  #　block的解析没有结束
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3  # 前一个卷积输出的通道个数（下一个卷积层的输入）
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # 卷积层
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)
            
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("bn_{0}".format(index), bn)
            
            if activation == "leaky":
                activa = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activa)
        
        # upsample
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(",")

            #start of a route
            start = int(x["layers"][0])

            #end ,if there exists one
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
            
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info,module_list
            
def predict_transform(prediction, inp_dim, anchors, num_classes, device):
    
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2) # 下采样倍数
    grid_size = inp_dim // stride  # 最后特征图的大小
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)  # 特征图上每个点的anchor box
    
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    
    # 预定义的anchor box进行缩放。如果是以原图的尺寸为基准设置的anchor box 
    # 就需要将其缩放到最后输出的特征图的尺寸上
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    
    # 对(x,y)偏移量以及置信度
    prediction[:,:, 0] = torch.sigmoid(prediction[:,:, 0])
    prediction[:,:, 1] = torch.sigmoid(prediction[:,:, 1])
    prediction[:,:, 4] = torch.sigmoid(prediction[:,:, 4])

    # 创建grid cell的中心坐标
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors).to(device)
    
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    
    prediction[:,:, 2:4] = torch.exp(prediction[:,:, 2:4]) * anchors
    
    prediction[:,:, 5:5 + num_classes] = torch.sigmoid((prediction[:,:, 5:5 + num_classes]))
    
    # 将bbox的尺寸放大到原图上
    prediction[:,:,:4] *= stride

    return prediction

def get_test_input():

    img = cv2.imread("person.jpg")
    img = cv2.resize(img, (416, 416))
    img = img[:,:,::-1].transpose(2, 0, 1)
    img = img[np.newaxis,:,:,:] / 255.0
    img = torch.from_numpy(img).float()
    return img

if __name__ == "__main__":
    
    anchors = generate_anchors()
    print(anchors)

    blocks = parse_cfg("yolov3.cfg")
    print(create_modules(blocks))