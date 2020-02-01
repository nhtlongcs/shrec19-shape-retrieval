from torch import nn
from torchvision.models import resnet50

# for test
from PIL import Image
from torchvision import transforms
from torch import cuda, flatten
import numpy as np
import torch
__all__ = ['ResNet']


class ResNet(nn.Module):
    """Some Information about ResNet"""

    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.model = resnet50(pretrained=pretrained)

        for param in self.model.parameters():
            param.requires_grad_(False)

    def _init_weights(self):
        if (self.pretrained):
            return
        for module in self.modules():
            # if isinstance(module, Up):
            module._init_weights()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = flatten(x, 1)
        return x


if __name__ == "__main__":

    model = ResNet()

    filename = '/home/ken/scripts blender/example/data/output/ring0/0/plane/render/Image0001.png'

    input_image = Image.open(filename)
    input_image = input_image.convert('RGB')
    print(np.array(input_image).shape)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        print(output[0].size())
