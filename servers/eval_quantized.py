"""
MobileNet model with quantization-aware training (QAT) enabled. Trained on CIFAR10.

This file batches training and evaluation into the same script. Since QAT requires careful
management of the training loop, it's easiest do both in the same run.
"""
# Forked from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py

####################
# MODEL DEFINITION #
####################

import torch.nn as nn
import torch.quantization
import math
from collections import OrderedDict
import time

# NOTE(aleksey): assigning layers names makes them easier to reference in the fuse_module code later
# on. fuse_modules takes a list of lists of layers as input, without named layers we'd have to use
# something like ['features.0.1', 'features.0.2']. Needless to say, that's not exactly readable.

def conv_bn(inp, oup, stride):
    return nn.Sequential(OrderedDict([
        ('q', torch.quantization.QuantStub()),
        ('conv2d', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
        ('batchnorm2d', nn.BatchNorm2d(oup)),
        ('relu6', nn.ReLU6(inplace=True)),
        ('dq', torch.quantization.DeQuantStub())        
    ]))

def conv_1x1_bn(inp, oup):
    return nn.Sequential(OrderedDict([
        ('q', torch.quantization.QuantStub()),
        ('conv2d', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        ('batchnorm2d', nn.BatchNorm2d(oup)),
        ('relu6', nn.ReLU6(inplace=True)),
        ('dq', torch.quantization.DeQuantStub())
    ]))

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(OrderedDict([
                ('q', torch.quantization.QuantStub()),
                # dw
                ('conv2d_1', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                ('bnorm_2', nn.BatchNorm2d(hidden_dim)),
                ('relu6_3', nn.ReLU6(inplace=True)),
                # pw-linear
                ('conv2d_4', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                ('bnorm_5', nn.BatchNorm2d(oup)),
                ('dq', torch.quantization.DeQuantStub())
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict([
                ('q', torch.quantization.QuantStub()),
                # pw
                ('conv2d_1', nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                ('bnorm_2', nn.BatchNorm2d(hidden_dim)),
                ('relu6_3', nn.ReLU6(inplace=True)),
                # dw
                ('conv2d_4', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                ('bnorm_5', nn.BatchNorm2d(hidden_dim)),
                ('relu6_6', nn.ReLU6(inplace=True)),
                # pw-linear
                ('conv2d_7', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                ('bnorm_8', nn.BatchNorm2d(oup)),
                ('dq', torch.quantization.DeQuantStub())
            ]))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        submodule_names = [
            'in_conv',
            *[f'inv_conv_{i}' for i in range(1, 18)],
            'out_conv'
        ]
        self.features = nn.Sequential(OrderedDict(list(zip(submodule_names, self.features))))

        # building classifier
        # NOTE(aleksey): setting qconfig to None disables quantization for this layer
        self.classifier = nn.Linear(self.last_channel, n_class)
        self.classifier.qconfig = None

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_model():
    return MobileNetV2(width_mult=1, n_class=10, input_size=32)


############
# TRAINING #
############

from torch import optim
import numpy as np

import torchvision
from torch.utils.data import DataLoader
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomPerspective(),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10("/mnt/cifar10/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def prepare_model(model):
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.fuse_modules(
        model,
        [
            # NOTE(aleksey): 'features' is the attr containing the non-head layers.
            ['features.in_conv.conv2d', 'features.in_conv.batchnorm2d'],
            ['features.inv_conv_1.conv.conv2d_1', 'features.inv_conv_1.conv.bnorm_2'],
            ['features.inv_conv_1.conv.conv2d_4', 'features.inv_conv_1.conv.bnorm_5'],
            *[
                *[[f'features.inv_conv_{i}.conv.conv2d_1',
                   f'features.inv_conv_{i}.conv.bnorm_2'] for i in range(2, 18)],
                *[[f'features.inv_conv_{i}.conv.conv2d_4',
                   f'features.inv_conv_{i}.conv.bnorm_5'] for i in range(2, 18)],
                *[[f'features.inv_conv_{i}.conv.conv2d_7',
                   f'features.inv_conv_{i}.conv.bnorm_8'] for i in range(2, 18)]
            ]
        ]
    )
    model = torch.quantization.prepare_qat(model)
    return model


def train(model):
    print(f"Training the model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    start_time = time.time()
    
    NUM_EPOCHS = 10
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for i, (X_batch, y_cls) in enumerate(dataloader):
            optimizer.zero_grad()

            y = y_cls
            X_batch = X_batch
            # y = y_cls.cuda()
            # X_batch = X_batch.cuda()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if i % 200 == 0:
                print(
                    f'Finished epoch {epoch}/{NUM_EPOCHS}, batch {i}. Loss: {curr_loss:.3f}.'
                )

            losses.append(curr_loss)

        print(
            f'Finished epoch {epoch}. '
            f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
        )
    print(f"Training done in {str(time.time() - start_time)} seconds.")


def eval_fn(model):
    model.eval()
    
    print(f"Converting the model (post-training)...")
    start_time = time.time()
    model = torch.quantization.convert(model)
    print(f"Quantization done in {str(time.time() - start_time)} seconds.")
    
    print(f"Evaluating the model...")
    start_time = time.time()
    for i, (X_batch, y_cls) in enumerate(dataloader):
        y = y_cls
        y_pred = model(X_batch)
    print(f"Evaluation done in {str(time.time() - start_time)} seconds.")

    print(f"Writing quantized model to disk.")
    torch.save(model.state_dict(), f'/spell/model_quantized.pth')

if __name__ == "__main__":
    model = get_model()
    model = prepare_model(model)
    train(model)
    eval_fn(model)
