
import numpy as np
import torch
import torch.nn as nn

from utils import parse_cfg,create_modules


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(blocks)
        
    
    def forward(self, x, device):
        modules = self.blocks[1:] 
        outputs = []

        write = 0
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
                    x = output[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = output[i + layers[0]]
                    map2 = output[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i + from_]