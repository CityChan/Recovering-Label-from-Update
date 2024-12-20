import torch
import torchvision
import torch.nn as nn
# from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor


# Models
# Only for MNIST
class DNN(nn.Module):
    def __init__(self, n_hidden=1, dim=300):
        super(DNN, self).__init__()
        self.fin = nn.Sequential(nn.Linear(784, dim), nn.ReLU())
        self.hiddens = []
        for i in range(n_hidden):
            self.hiddens.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))
        self.hiddens = nn.Sequential(*self.hiddens)
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = self.fin(x)
        x = self.hiddens(x)
        y = self.fc(x)
        return y, x


class LeNet5(nn.Module):
    # MNIST:1,256,10; CIFAR-10:3,400,10; CIFAR-100:3,400,100;
    def __init__(self, channel=1, hidden=256, num_classes=10, ns=0.01, bn=False, tanh=False, leaky_relu=False):
        super(LeNet5, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(channel, 6, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(6) if bn else nn.Identity(),
                                    nn.Tanh() if tanh else nn.LeakyReLU(negative_slope=ns) if leaky_relu else nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(16) if bn else nn.Identity(),
                                    nn.Tanh() if tanh else nn.LeakyReLU(negative_slope=ns) if leaky_relu else nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.nlt1 = nn.Sequential(nn.Linear(hidden, 120),
                                  nn.Tanh() if tanh else nn.LeakyReLU(negative_slope=ns) if leaky_relu else nn.ReLU())
        self.nlt2 = nn.Sequential(nn.Linear(120, 84),
                                  nn.Tanh() if tanh else nn.LeakyReLU(negative_slope=ns) if leaky_relu else nn.ReLU())
        self.fc = torch.nn.Linear(84, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        emb = out.view(out.size(0), -1)
        emb = self.nlt1(emb)
        emb = self.nlt2(emb)
        out = self.fc(emb)
        return out, emb


class LeNetZhu(nn.Module):
    # MNIST:1,588,10; CIFAR-10:3,768,10; CIFAR-100:3,768,100;
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super(LeNetZhu, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )
        for m in self.modules():
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
            if hasattr(m, "bias"):
                m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        emb = out.view(out.size(0), -1)
        out = self.fc(emb)
        return out, emb


class VGG(nn.Module):

    def __init__(self, block_nums, hidden, num_classes=10, dropout=True):
        super(VGG, self).__init__()
        self.block1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.block2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.block3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.block4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.block5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])
        self.exactor = nn.Sequential(
            nn.Linear(hidden, 4096),
            nn.ReLU(inplace=True),
#             nn.ELU(inplace=True),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
#             nn.ELU(inplace=True),
            nn.Dropout(0.5) if dropout else nn.Identity()
        )
        self.fc = nn.Linear(4096, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(in_channels, out_channels, block_num):
        blocks = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
#             nn.ELU(inplace=True)
        )]
        for i in range(1, block_num):
            blocks.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
#                 nn.ELU(inplace=True)
            ))
        blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, start_dim=1)
        emb = self.exactor(x)
        out = self.fc(emb)
        return out, emb


    
#------------------------------------------------------------------------------------------------------------
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )




class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    


#----------------------------------------------------------------------------------
# ResNet implementation from https://github.com/JonasGeiping/invertinggradients

# torchvision.models.ResNet
class ResNet(nn.Module):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, channel=3, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
        """Initialize as usual. Layers and strides are scriptable."""
        super(ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(channel, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.SiLU(inplace=True)
    
        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(
                self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)

        return y, x

    def forward(self, x: Tensor) -> Tensor:
            return self._forward_impl(x)
        
        
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
def get_model(model_name,
              net_params,
              device,
              n_hidden=1,
              n_dim=300,
              batchnorm=False,
              dropout=False,
              tanh=False,
              leaky_relu=False):
    num_classes, channel, hidden = net_params
    if model_name == 'dnn':
        model = DNN(n_hidden=n_hidden, dim=n_dim)
    elif model_name == 'lenet5':
        model = LeNet5(channel=channel, hidden=hidden, num_classes=num_classes, bn=batchnorm, tanh=tanh,
                       leaky_relu=leaky_relu)
    elif model_name == 'lenetzhu':
        model = LeNetZhu(channel=channel, hidden=hidden, num_classes=num_classes)
    elif model_name == 'vgg11':
        model = VGG([1, 1, 2, 2, 2], hidden=hidden, num_classes=num_classes, dropout=dropout)
    elif model_name == 'vgg13':
        model = VGG([2, 2, 2, 2, 2], hidden=hidden, num_classes=num_classes, dropout=dropout)
    elif model_name == 'vgg16':
        model = VGG([2, 2, 3, 3, 3], hidden=hidden, num_classes=num_classes, dropout=dropout)
    elif model_name == 'vgg19':
        model = VGG([2, 2, 4, 4, 4], hidden=hidden, num_classes=num_classes, dropout=dropout)
    elif model_name == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2],
                       num_classes=num_classes, base_width=64, channel=channel)
    elif model_name == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3],
                       num_classes=num_classes, base_width=64, channel=channel)
    elif model_name == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3],
                       num_classes=num_classes, base_width=64, channel=channel)
    elif model_name == 'resnet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3],
                       num_classes=num_classes, base_width=64, channel=channel)
    elif model_name == 'resnet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3],
                       num_classes=num_classes, base_width=64, channel=channel)
    else:
        raise NotImplementedError('Model not implemented.')
    model = model.to(device)
    return model
