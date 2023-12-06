import os
import pdb
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .detection_head import DetectionHead
from .sra_head import SRAHead
import time
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

BatchNorm = nn.BatchNorm2d


__all__ = ['drn_a_18']


webroot = 'http://dl.yf.io/drn/'
base_url = 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/'
model_urls = {
    'drn_a_50': base_url + 'resnet50-imagenet.pth',
    'drn_a_18': base_url + 'resnet18-imagenet.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3_d(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None,
#                  dilation=(1, 1), residual=True):
#         super(BasicBlock, self).__init__()
#         # self.conv1 = conv3x3(inplanes, planes, stride)
#         self.conv1 = conv3x3_d(inplanes, planes, stride,
#                              padding=dilation[0], dilation=dilation[0])
#         self.bn1 = BatchNorm(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3_d(planes, planes,
#                              padding=dilation[1], dilation=dilation[1])
#         self.bn2 = BatchNorm(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.residual = residual
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         if self.residual:
#             out += residual
#         out = self.relu(out)
#
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        self.layer5_1 = self._make_layer2(block, 256, layers[4], stride=2)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
            self.layer8_1 = self._make_layer2(block, 512, layers[7], stride=2)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        # y.append(x)
        x = self.layer2(x)
        # y.append(x)
        #
        x = self.layer3(x)
        y.append(x)
        #
        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        x = self.layer5_1(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            # y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            # y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            x = self.layer8_1(x)
            y.append(x)

        return tuple(y)
        # if self.out_map:
        #     x = self.fc(x)
        # else:
        #     x = self.avgpool(x)
        #     x = self.fc(x)
        #     x = x.view(x.size(0), -1)
        #
        # if self.out_middle:
        #     return x, y
        # else:
        #     return x
class Convkxk(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1,
                 padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPEM(nn.Module):
    def __init__(self, planes):
        super(FPEM, self).__init__()
        self.dwconv3_1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer3_1 = Convkxk(planes, planes)

        self.dwconv2_1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer2_1 = Convkxk(planes, planes)

        self.dwconv1_1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer1_1 = Convkxk(planes, planes)

        self.dwconv2_2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer2_2 = Convkxk(planes, planes)

        self.dwconv3_2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer3_2 = Convkxk(planes, planes)

        self.dwconv4_2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias=False)
        self.smooth_layer4_2 = Convkxk(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))

        f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_, f1_)))
        f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_, f2_)))
        f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))

        f1 = f1 + f1_
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4 + f4_

        return f1, f2, f3, f4


class DRN_A(nn.Module):

    def __init__(self, block, layers, num_classes=6, scale=1, rec_cfg=None, rec_cscale=1):

        self.inplanes = 128
        super(DRN_A, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.out_dim = 512 * block.expansion

        self.layer1 = self._make_layer2(block, 64, layers[0])
        self.layer2 = self._make_layer2(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer2(block, 256, layers[2], stride=2)
        # self.layer3_1 = self._make_layer2(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer2(block, 512, layers[3], stride=2)

        hidden_dim = 128

        # reduce dim layer
        self.reduce_layer4 = Convkxk(512 * block.expansion, hidden_dim)
        self.reduce_layer3 = Convkxk(256 * block.expansion, hidden_dim)
        self.reduce_layer2 = Convkxk(128 * block.expansion, hidden_dim)
        self.reduce_layer1 = Convkxk(64 * block.expansion, hidden_dim)

        # FPEM
        self.fpem1 = FPEM(hidden_dim)
        self.fpem2 = FPEM(hidden_dim)

        # detection head
        self.detection_head = DetectionHead(
            hidden_dim * 4,
            hidden_dim,
            num_classes)

        self.recognition_head = None
        if rec_cfg is not None:
            rec_cfg['input_dim'] = 512
            rec_cfg['hidden_dim'] = int(128 * rec_cscale)
            self.recognition_head = SRAHead(**rec_cfg)

        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                args=None):
        if args.report_speed:
            time_cost = {}
            torch.cuda.synchronize()
            start = time.time()

        h = imgs
        h = self.relu1(self.bn1(self.conv1(h)))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.relu3(self.bn3(self.conv3(h)))
        h = self.maxpool(h)

        h = self.layer1(h)
        f1 = self.reduce_layer1(h)

        h = self.layer2(h)
        f2 = self.reduce_layer2(h)

        h = self.layer3(h)
        f3 = self.reduce_layer3(h)

        h = self.layer4(h)
        f4 = self.reduce_layer4(h)

        if args.report_speed:
            torch.cuda.synchronize()
            time_cost['backbone_time'] = time.time() - start
            start = time.time()

        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)

        f2 = self._upsample(f2, f1)
        f3 = self._upsample(f3, f1)
        f4 = self._upsample(f4, f1)
        f = torch.cat((f1, f2, f3, f4), 1)

        if args.report_speed:
            torch.cuda.synchronize()
            time_cost['neck_time'] = time.time() - start
            start = time.time()

        out_det = self.detection_head(
            f,
            (imgs.size(2) // self.scale, imgs.size(3) // self.scale))

        if args.report_speed:
            torch.cuda.synchronize()
            time_cost['det_head_time'] = time.time() - start
            start = time.time()

        outputs = {}
        if self.training:
            loss_det = self.detection_head.loss(
                out_det,
                gt_texts,
                gt_kernels,
                training_masks,
                gt_instances,
                gt_bboxes, args=args)
            outputs.update(loss_det)
        else:
            res_det = self.detection_head.get_results(imgs, out_det, args=args)
            outputs.update(res_det)

        if self.recognition_head is not None:
            if self.training:
                x_crops, gt_words = self.recognition_head.extract_feature(
                    f, (imgs.size(2), imgs.size(3)),
                    gt_instances * training_masks, gt_bboxes, gt_words,
                    word_masks)
                if x_crops is not None:
                    out_rec = self.recognition_head(x_crops, gt_words)
                    loss_rec = self.recognition_head.loss(
                        out_rec,
                        gt_words,
                        reduce=False)
                else:
                    loss_rec = {
                        'loss_rec': f.new_full((1,), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1,), -1, dtype=torch.float32)
                    }
                outputs.update(loss_rec)
            else:
                if args.report_speed:
                    torch.cuda.synchronize()
                    start = time.time()
                if len(res_det['bboxes']) > 0:
                    x_crops = self.recognition_head.extract_feature_test(
                        f,
                        (imgs.size(2), imgs.size(3)),
                        f.new_tensor(
                            res_det['label'],
                            dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(
                            res_det['bboxes_h'],
                            dtype=torch.long),
                        unique_labels=res_det['instances'])
                    words, word_scores = self.recognition_head.forward(
                        x_crops,
                        args=args)
                else:
                    words = []
                    word_scores = []
                    if args.report_speed:
                        torch.cuda.synchronize()
                        start = time.time()

                if args.report_speed:
                    torch.cuda.synchronize()
                    time_cost['rec_time'] = time.time() - start

                outputs['words'] = words
                outputs['word_scores'] = word_scores
                outputs['label'] = ''

        if args.report_speed:
            outputs.update(time_cost)

        return outputs


def drn_a_18(pretrained=False, **kwargs):
    model = DRN_A(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['drn_a_18']), strict=False)
    return model


def drn_a_50(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['drn_a_50']), strict=False)
    return model


def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['drn-c-26']), strict=False)
    return model


def drn_c_42(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
    return model


def drn_c_58(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
    return model


def drn_d_22(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model


def drn_d_24(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-24']))
    return model


def drn_d_38(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-38']))
    return model


def drn_d_40(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-40']))
    return model


def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model


def drn_d_56(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-56']))
    return model


def drn_d_105(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model


def drn_d_107(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-107']))
    return


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)