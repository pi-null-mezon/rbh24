import torchvision.models
from torch import nn
from collections import OrderedDict

# Feature extraction points according
# [A Systematic Performance Analysis of Deep Perceptual Loss Networks:
# Breaking Transfer Learning Conventions](https://arxiv.org/pdf/2302.04032.pdf)
def get_discriminator_network(name):
    if name == 'vgg11':
        model = torchvision.models.vgg11(num_classes=1)
        weights = torchvision.models.VGG11_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)
        weights['classifier.6.weight'] = weights['classifier.6.weight'][:1, :]
        weights['classifier.6.bias'] = weights['classifier.6.bias'][:1]
    elif name == 'resnet18':
        model = torchvision.models.resnet18(num_classes=1)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)
        weights['fc.weight'] = weights['fc.weight'][:1, :]
        weights['fc.bias'] = weights['fc.bias'][:1]
    elif name == 'effnetb0':
        model = torchvision.models.efficientnet_b0(num_classes=1)
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)
        weights['classifier.1.weight'] = weights['classifier.1.weight'][:1, :]
        weights['classifier.1.bias'] = weights['classifier.1.bias'][:1]
    elif name == 'tiny':
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, 3, 2, padding=1)),
            ('bn1', nn.BatchNorm2d(16)),            
            ('prelu1', nn.PReLU(16)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(16, 32, 3, 2, padding=1)),
            ('bn2', nn.BatchNorm2d(32)),            
            ('prelu2', nn.PReLU(32)),
            ('conv3', nn.Conv2d(32, 64, 3, 2, padding=1)),
            ('bn3', nn.BatchNorm2d(64)),            
            ('prelu3', nn.PReLU(64)),
            ('conv4', nn.Conv2d(64, 32, 3, 2, padding=1)),
            ('bn4', nn.BatchNorm2d(32)),            
            ('prelu4', nn.PReLU(32)),
            ('conv5', nn.Conv2d(32, 16, 3, 2, padding=1)),
            ('bn5', nn.BatchNorm2d(16)),            
            ('prelu5', nn.PReLU(16)),
            ('conv6', nn.Conv2d(16, 8, 3, 1, padding=1)),
            ('bn6', nn.BatchNorm2d(8)),            
            ('prelu6', nn.PReLU(8)),
            ('conv7', nn.Conv2d(8, 1, 3, 1, padding=1)),
            ('bn7', nn.BatchNorm2d(1)),            
            ('pool7', nn.AdaptiveAvgPool2d(output_size=(1, 1))),
            ('flatten', nn.Flatten()),
        ]))
        weights = None

    if weights:
        model.load_state_dict(weights)
   
    return model


if __name__ == "__main__":
    p = get_discriminator_network('tiny')
    import torch
    print(p(torch.randn(2, 3, 128, 128)).shape)
