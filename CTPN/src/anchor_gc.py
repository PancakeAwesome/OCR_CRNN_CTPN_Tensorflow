import numpy as np


class AnchorText:
    """docstring for AnchorText"""
    def __init__(self):
        # feature map上的每个像素点对应10个固定宽度不同高度的anchors
        self.anchor_num = 10
