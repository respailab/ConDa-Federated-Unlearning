import torch.nn as nn
from torchvision.models import resnet18, resnet50


# Added required classes
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            #             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            #                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, filters_percentage=1, dropout=False, batch_norm=True):
        super().__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)

        self.conv1 = Conv(num_channels, n_filter1,
                          kernel_size=3, batch_norm=batch_norm)
        self.conv2 = Conv(n_filter1, n_filter1,
                          kernel_size=3, batch_norm=batch_norm)
        self.conv3 = Conv(n_filter1, n_filter2, kernel_size=3,
                          stride=2, padding=1, batch_norm=batch_norm)

        self.dropout1 = self.features = nn.Sequential(
            nn.Dropout(inplace=True) if dropout else Identity())

        self.conv4 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=1, batch_norm=batch_norm)
        self.conv5 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=1, batch_norm=batch_norm)
        self.conv6 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=2, padding=1, batch_norm=batch_norm)

        self.dropout2 = self.features = nn.Sequential(
            nn.Dropout(inplace=True) if dropout else Identity())

        self.conv7 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=1, batch_norm=batch_norm)
        self.conv8 = Conv(n_filter2, n_filter2, kernel_size=1,
                          stride=1, batch_norm=batch_norm)
        # self.pool = nn.AvgPool2d(7)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(n_filter2*2*2, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        out = self.dropout1(out)

        out = self.conv4(out)

        out = self.conv5(out)

        out = self.conv6(out)

        out = self.dropout2(out)

        out = self.conv7(out)

        out = self.conv8(out)

        out = self.pool(out)

        out = self.flatten(out)

        out = self.classifier(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, pretrained=False):
        super().__init__()
        base = resnet18(pretrained=pretrained)
        self.base = nn.Sequential(*list(base.children())[:-1])
        if pretrained:
            for param in self.base.parameters():
                param.requires_grad = False
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        x = self.final(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        base = resnet50(pretrained=pretrained)
        self.base = nn.Sequential(*list(base.children())[:-1])
        if pretrained:
            for param in self.base.parameters():
                param.requires_grad = False
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        x = self.final(x)
        return x
