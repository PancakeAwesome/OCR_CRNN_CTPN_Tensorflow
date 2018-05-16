import cv2, caffe
import numpy as np
from matplotlib import cm

def prepare_im(im, mean):
    """图片均值化并transose成[batch_size, height, width]的形式"""
    im_data = np.transpose(im - mean, (2, 0, 1))
    return im_data

def resize_im(im, scale, max_scale=None):
    """
    缩放图片
    """
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f), f

def normalize(data):
    """
    数据归一化
    """
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_

def draw_boxes(im, bboxes, is_display=True, color=None, caption="Image", wait=True):
    """
        在图片上做bbox的label，并返回text boxes的数据和打上text boxes的图片
        boxes: bounding boxes
    """
    text_recs=np.zeros((len(bboxes), 8), np.int)

    im=im.copy()
    index = 0
    for box in bboxes:
        if color==None:
            if len(box)==8 or len(box)==9:
                c=tuple(cm.jet([box[-1]])[0, 2::-1]*255)
            else:
                c=tuple(np.random.randint(0, 256, 3))
        else:
            c=color
        
        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2
        
        disX = x2 - x1
        disY = y2 - y1
    width = np.sqrt(disX*disX + disY*disY)
    fTmp0 = y3 - y1
    fTmp1 = fTmp0 * disY / width
    x = np.fabs(fTmp1*disX / width)
    y = np.fabs(fTmp1*disY / width)
        if box[5] < 0:
           x1 -= x
           y1 += y
           x4 += x
           y4 -= y
        else:
           x2 += x
           y2 += y
           x3 -= x
           y3 -= y
        cv2.line(im,(int(x1),int(y1)),(int(x2),int(y2)),c,2)
        cv2.line(im,(int(x1),int(y1)),(int(x3),int(y3)),c,2)
        cv2.line(im,(int(x4),int(y4)),(int(x2),int(y2)),c,2)
        cv2.line(im,(int(x3),int(y3)),(int(x4),int(y4)),c,2)
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1
        #cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c,2)  
    if is_display:
        cv2.imshow('result', im)
        #if wait:
            #cv2.waitKey(0)
    return text_recs

class CaffeModel:
    """docstring for CaffeModel"""
    def __init__(self, net_def_file, model_file):
        self.net_def_file = net_def_file
        # 通过net文件和模型文件形成caffe模型
        self.net = caffe.Net(net_def_file, model_file, caffe.TEST) 

    def blob(self, key):
        """返回每层的输出结果"""
        return self.net.blobs[key].data.copy()

    # data参数是一个np数组
    def forward(self, input_data):
        return self.forward2({
            "data": input_data[np.newaxis, :]
            })

    # data参数是一个字典
    def forward2(self, input_data):
        for k, v in input_data.items():
            self.net.blobs[k].reshape(*v.shape)
            # [...]是深拷贝等同于[:]
            self.net.blobs[k].data[...] = v
        return self.net.forward()

    def net_def_file(self):
        return self.net_def_file






