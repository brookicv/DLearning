
import numpy as np 


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


if __name__ == "__main__":
    
    anchors = generate_anchors()
    print(anchors)