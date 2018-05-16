from other import clip_boxes
from anchor import AnchorText


class ProposalLayer(caffe.layer):
    """proposal层"""
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        # 读取net类中的参数
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        # 分配anchor构造器属性
        self.anchor_generator = AnchorText()
        self._num_anchors = self.anchor_generator.anchor_num

        top[0].reshape(1, 4)
        top[1].reshape(1, 1, 1, 1)
        