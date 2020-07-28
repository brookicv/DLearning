
import numpy as np
import torch
import torch.nn as nn

from utils import parse_cfg,create_modules,predict_transform


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

        self.anchors_19 = torch.FloatTensor([[3.6250, 2.8125], [4.8750, 6.1875],[11.6562, 10.1875]])
        self.anchors_38 = torch.FloatTensor([[1.8750, 3.8125], [3.8750, 2.8125],[3.6875, 7.4375]])
        self.anchors_76 = torch.FloatTensor([[1.2500, 1.6250],[2.0000, 3.7500],[4.1250, 2.8750]])
    
    def create_grid(self, grid_size,i):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        anchors = self.module_list[i][0].anchors
        inp_dim = int(self.net_info["height"])
        stride = inp_dim // grid_size  # 下采样倍数

        num_anchors = len(anchors)  # 特征图上每个点的anchor box
         # 创建grid cell的中心坐标
        grid = torch.arange(grid_size)
        a, b = torch.meshgrid(grid, grid) # a是y轴的坐标，b是x轴的坐标

        x_offset = b.float().view(-1, 1)
        y_offset = a.float().view(-1, 1)

        x_offset = x_offset.to(device)
        y_offset = y_offset.to(device)

        self.x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

         # 预定义的anchor box进行缩放。如果是以原图的尺寸为基准设置的anchor box 
        # 就需要将其缩放到最后输出的特征图的尺寸上
        anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

        anchors = torch.FloatTensor(anchors).to(device)
        if i == 82:
            self.anchors = self.anchors_19.repeat(grid_size * grid_size, 1).unsqueeze(0).to(device)
        elif i == 94:
            self.anchors = self.anchors_38.repeat(grid_size * grid_size, 1).unsqueeze(0).to(device)
        elif i == 106:
            self.anchors = self.anchors_76.repeat(grid_size * grid_size, 1).unsqueeze(0).to(device)

        self.grid_size = grid_size

    def forward(self, x):
        modules = self.blocks[1:] 
        outputs = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        write = 0
        preds = []
        # block和module_list的module是一一对应的
        #　通过block判断其类型，然后调用相应的方法
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
                
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                prediction = x
                batch_size = prediction.size(0)
                stride = inp_dim // prediction.size(2) # 下采样倍数
                grid_size = inp_dim // stride  # 最后特征图的大小
                bbox_attrs = 5 + num_classes
                num_anchors = len(anchors)  # 特征图上每个点的anchor box

                self.create_grid(grid_size,i)
                
                prediction = prediction.view(batch_size, bbox_attrs * num_anchors, self.grid_size * self.grid_size)
                prediction = prediction.transpose(1, 2).contiguous()
                prediction = prediction.view(batch_size, self.grid_size * self.grid_size * 3, 85)
                
                
                # 对(x,y)偏移量以及置信度
                # 将bbox的尺寸放大到原图上
                xy = (torch.sigmoid(prediction[:,:,0:2]) + self.x_y_offset.float()) 
                wh = (torch.exp(prediction[:,:, 2:4]) * self.anchors.float())

                # xy = xy * float(stride)
                # wh = wh * float(stride)

                confidence = torch.sigmoid(prediction[:,:,4:5])
                p_cls = torch.sigmoid((prediction[:,:, 5:85]))
                
                preds.append(torch.cat((xy,wh,confidence,p_cls),dim=2))
                

                """
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, device)
                
                if not write:
                    detection = x # 有三个detection的输出，第一个直接赋值
                    write = 1
                else:
                    detection = torch.cat((detection, x), 1)
                """
                    
            outputs[i] = x

        preds = torch.cat(preds, 1)
        return preds[:,:,:5],preds[:,:,5:5+ num_classes]
        # return preds

    def load_weights(self, weightfile):
        
        fp = open(weightfile, "rb")
        
        # 前5个值表示header
        # 1. Major version numbers
        # 2. Minor version numbers
        # 3. Subversion numbers
        # 4. 5. Image seen by the network(during training)

        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        # 余下部分为权重，权重以float32的形式保存
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1] # batch normalization layer

                    num_bn_biases = bn.bias.numel()  # 得到权重的个数(bn layer)
                    
                    # load weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:

                    # bias个数计算
                    num_biases = conv.bias.numel()

                    # load weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

