import torch.nn as nn
import utils


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, ngpu):
        super(BidirectionalLSTM, self).__init__()
        self.ngpu = ngpu

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = utils.data_parallel(
            self.rnn, input, self.ngpu)  # [T, b, h * 2]

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = utils.data_parallel(
            self.embedding, t_rec, self.ngpu)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    """crnn模型"""
    def __init__(self, imgH, nc, nclass, nh, ngpu, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        self.ngpu = ngpu
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2),
                                                            (2, 1),
                                                            (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2),
                                                            (2, 1),
                                                            (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh, ngpu),
            BidirectionalLSTM(nh, nh, nclass, ngpu)
        )

    def forward(self, input):
        # conv features
        conv = utils.data_parallel(self.cnn, input, self.ngpu)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = utils.data_parallel(self.rnn, conv, self.ngpu)

        return output