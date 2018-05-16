from cfg import Config as cfg
from other import prepare_img, normalize
import numpy as np
from utils.cpu_nms import cpu_nms as nms
from text_proposal_connector import TextProposalConnector

class TextProposalDetector:
    """TextDetector的子类"""
    def __init__(self, caffe_model):
        self.caffe_model = caffe_model

    def detect(self, im, mean):
        """
        通过ctpn网络初步形成proposals
        """
        im_data = prepare_img(im, mean)
        # caffe的前向传播
        # 参数为前向传播必须的数据
        # data,im_info
        _ =  self.caffe_model.forward2({
            "data": im_data[np.newaxis, :],
            "im_info": np.array([[im_data.shape[1], im_data.shape[2]]], np.float32)
            })
        # 得到caffe模型的proposal层的输出结果
        rois = self.caffe_model.blob("rois")
        scores = self.caffe_model.blob("scores")
        return rois, scores

class TextDetector:
    """docstring for TextDetector"""
    def __init__(self, text_proposal_detector):
        self.text_proposal_detector = text_proposal_detector
        self.text_proposal_connector = TextProposalConnector()

    def detect(self, im):
        """
        Detecting text boxes from an image
        :return: the bounding boxes of the detected texts
        """
        text_proposals, scores = self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals = text_proposals[keep_inds]
        scores = scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[keep_inds]

        # nms for text propoasls
        keep_inds = nms(np.hstack((text_proposals, scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        # 得分归一化统一为[0,1]之间
        scores = normalize(scores)

        text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, im.shape[:2])
        # text_lines:(x1, x2, y1, y2, score)

        # 筛选text boxes
        keep_inds = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_inds]

        if text_lines.shape[0] != 0:
            keep_inds = nms(text_lines, cfg.TETEXT_LINE_NMS_THRESHXL)
            text_lines = text_lines[keep_inds]

        return text_lines

    def filter_boxes(self, boxes):
        heights = boxes[:, 3] - boxes[:, 1] + 1
        width = boxes[: 2] - boxes[: 0] + 1
        scores = boxes[:, -1]
        return np.where((widths/heights>cfg.MIN_RATIO) & (scores>cfg.LINE_MIN_SCORE) &
                          (widths>(cfg.TEXT_PROPOSALS_WIDTH*cfg.MIN_NUM_PROPOSALS)))[0]
        