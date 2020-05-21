import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing
import mxnet.gluon as gluon
import mxnet as mx
import numpy as np
import random
import time
import cv2
import os

from mxnet import nd, autograd, gluon
from mxnet.image import resize_short
from mxboard import SummaryWriter
from mxnet.gluon.model_zoo.vision import resnet34_v1
from mxnet.contrib.ndarray import MultiBoxPrior, MultiBoxTarget, MultiBoxDetection, box_nms
from skimage.draw import line_aa

class SmoothL1Loss(gluon.loss.Loss):
    '''
    A SmoothL1loss function defined in https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html
    '''
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)

class SSD(gluon.Block):
    def __init__(self, num_classes, ctx, **kwargs):
        super(SSD, self).__init__(**kwargs)

        # Seven sets of anchor boxes are defined. For each set, n=2 sizes and m=3 ratios are defined.
        # Four anchor boxes (n + m - 1) are generated: 2 square anchor boxes based on the n=2 sizes and 2 rectanges based on
        # the sizes and the ratios. See https://discuss.mxnet.io/t/question-regarding-ssd-algorithm/1307 for more information.
        
        #self.anchor_sizes = [[.1, .2], [.2, .3], [.2, .4], [.4, .6], [.5, .7], [.6, .8], [.7, .9]]
        #self.anchor_ratios = [[1, 3, 5], [1, 3, 5], [1, 6, 8], [1, 5, 7], [1, 6, 8], [1, 7, 9], [1, 7, 10]]

        self.anchor_sizes = [[.1, .2], [.2, .3], [.2, .4], [.3, .4], [.3, .5], [.4, .6]]
        self.anchor_ratios = [[1, 3, 5], [1, 3, 5], [1, 6, 8], [1, 4, 7], [1, 6, 8], [1, 5, 7]]

        self.num_anchors = len(self.anchor_sizes)
        self.num_classes = num_classes
        self.ctx = ctx
        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = self.get_ssd_model()
            self.downsamples.initialize(mx.init.Normal(), ctx=self.ctx)
            self.class_preds.initialize(mx.init.Normal(), ctx=self.ctx)
            self.box_preds.initialize(mx.init.Normal(), ctx=self.ctx)

    def get_body(self):
        '''
        Create the feature extraction network of the SSD based on resnet34.
        The first layer of the res-net is converted into grayscale by averaging the weights of the 3 channels
        of the original resnet.

        Returns
        -------
        network: gluon.nn.HybridSequential
            The body network for feature extraction based on resnet
        
        '''
        pretrained = resnet34_v1(pretrained=True, ctx=self.ctx)
        pretrained_2 = resnet34_v1(pretrained=True, ctx=mx.cpu(0))
        first_weights = pretrained_2.features[0].weight.data().mean(axis=1).expand_dims(axis=1)
        # First weights could be replaced with individual channels.
        
        body = gluon.nn.HybridSequential()
        with body.name_scope():
            first_layer = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2), in_channels=1, use_bias=False)
            first_layer.initialize(mx.init.Normal(), ctx=self.ctx)
            first_layer.weight.set_data(first_weights)
            body.add(first_layer)
            body.add(*pretrained.features[1:-3])
        return body

    def get_class_predictor(self, num_anchors_predicted):
        '''
        Creates the category prediction network (takes input from each downsampled feature)

        Parameters
        ----------
        
        num_anchors_predicted: int
            Given n sizes and m ratios, the number of boxes predicted is n+m-1.
            e.g., sizes=[.1, .2], ratios=[1, 3, 5] the number of anchors predicted is 4.

        Returns
        -------

        network: gluon.nn.HybridSequential
            The class predictor network
        '''
        return gluon.nn.Conv2D(num_anchors_predicted*(self.num_classes + 1), kernel_size=3, padding=1)

    def get_box_predictor(self, num_anchors_predicted):
        '''
        Creates the bounding box prediction network (takes input from each downsampled feature)
        
        Parameters
        ----------
        
        num_anchors_predicted: int
            Given n sizes and m ratios, the number of boxes predicted is n+m-1.
            e.g., sizes=[.1, .2], ratios=[1, 3, 5] the number of anchors predicted is 4.

        Returns
        -------

        pred: gluon.nn.HybridSequential
            The box predictor network
        '''
        pred = gluon.nn.HybridSequential()
        with pred.name_scope():
            pred.add(gluon.nn.Conv2D(channels=num_anchors_predicted*4, kernel_size=3, padding=1))
        return pred

    def get_down_sampler(self, num_filters):
        '''
        Creates a two-stacked Conv-BatchNorm-Relu and then a pooling layer to
        downsample the image features by half.
        '''
        out = gluon.nn.HybridSequential()
        for _ in range(2):
            out.add(gluon.nn.Conv2D(num_filters, 3, strides=1, padding=1))
            out.add(gluon.nn.BatchNorm(in_channels=num_filters))
            out.add(gluon.nn.Activation('relu'))
        out.add(gluon.nn.MaxPool2D(2))
        out.hybridize()
        return out

    def get_ssd_model(self):
        '''
        Creates the SSD model that includes the image feature, downsample, category
        and bounding boxes prediction networks.
        '''
        body = self.get_body()
        downsamples = gluon.nn.HybridSequential()
        class_preds = gluon.nn.HybridSequential()
        box_preds = gluon.nn.HybridSequential()

        downsamples.add(self.get_down_sampler(32))
        downsamples.add(self.get_down_sampler(32))
        downsamples.add(self.get_down_sampler(32))

        for scale in range(self.num_anchors):
            num_anchors_predicted = len(self.anchor_sizes[0]) + len(self.anchor_ratios[0]) - 1
            class_preds.add(self.get_class_predictor(num_anchors_predicted))
            box_preds.add(self.get_box_predictor(num_anchors_predicted))

        return body, downsamples, class_preds, box_preds

    def ssd_forward(self, x):
        '''
        Helper function of the forward pass of the sdd
        '''
        x = self.body(x)

        default_anchors = []
        predicted_boxes = []
        predicted_classes = []

        for i in range(self.num_anchors):
            default_anchors.append(MultiBoxPrior(x, sizes=self.anchor_sizes[i], ratios=self.anchor_ratios[i]))
            predicted_boxes.append(self._flatten_prediction(self.box_preds[i](x)))
            predicted_classes.append(self._flatten_prediction(self.class_preds[i](x)))
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
            elif i == 3:
                x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))
        return default_anchors, predicted_classes, predicted_boxes

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = self.ssd_forward(x)
        # we want to concatenate anchors, class predictions, box predictions from different layers
        anchors = nd.concat(*default_anchors, dim=1)
        box_preds = nd.concat(*predicted_boxes, dim=1)
        class_preds = nd.concat(*predicted_classes, dim=1)
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))
        return anchors, class_preds, box_preds

    def _flatten_prediction(self, pred):
        '''
        Helper function to flatten the predicted bounding boxes and categories
        '''
        return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))

    def training_targets(self, default_anchors, class_predicts, labels):
        '''
        Helper function to obtain the bounding boxes from the anchors.
        '''
        class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
        box_target, box_mask, cls_target = MultiBoxTarget(default_anchors, labels, class_predicts)
        return box_target, box_mask, cls_target


if __name__ == "__main__":
    net = SSD(num_classes=2, ctx=ctx)
    net.hybridize()
    net.load_parameters("ssd_word400.params")