import torchvision.models
from torchvision.models.feature_extraction import create_feature_extractor


# Feature extraction points according
# [A Systematic Performance Analysis of Deep Perceptual Loss Networks:
# Breaking Transfer Learning Conventions](https://arxiv.org/pdf/2302.04032.pdf)
def get_perceptual_loss_network(name):
    if name == 'vgg11':
        model = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.IMAGENET1K_V1)
        return_nodes = {
            'features.1': 'relu_1',
            'features.4': 'relu_2',
            'features.9': 'relu_4',
            'features.19': 'relu_8',
        }
    elif name == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        return_nodes = {
            'relu': 'relu_1',
            'layer1': 'block_1',
            'layer2': 'block_2',
            'layer4': 'block_4',
        }
    elif name == 'effnetb0':
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        return_nodes = {
            'features.0': 'silu_1',
            'features.1.0.block': 'mbconv_1',
            'features.4.0.block': 'mbconv_4',
            'features.7.0.block': 'mbconv_7',
        }
    return create_feature_extractor(model, return_nodes=return_nodes)


if __name__ == "__main__":
    p = get_perceptual_loss_network('effnetb0')
