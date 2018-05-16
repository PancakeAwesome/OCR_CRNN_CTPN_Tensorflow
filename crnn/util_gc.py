#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
import collections


class strLabelConverter(object):
    """docstring for strLabelConverter"""
    def __init__(self, alphabet):
        # '-' 作为字母ending
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i, char in enumerate(alphabet):
            # index:0按照ctc的要求是blank
            self.dict[char] = i + 1
        