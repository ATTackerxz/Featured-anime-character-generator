import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    classname = w.__class__.__name__
    if (type(w) == nn.ConvTranspose2d or type(w) == nn.Conv2d):
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif (type(w) == nn.BatchNorm2d):
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)
    elif (type(w) == nn.Linear):
        nn.init.normal_(w.weight.data, 0.0, 0.02)

# Define the Generator Network
class Generator_(nn.Module):
    def __init__(self, params):
        super(Generator_, self).__init__()

        self.fc_embed1 = nn.Linear(params['lsize'], 128, bias=False)

        # Input is the latent vector Z + Conditions.
        self.tconv1 = nn.ConvTranspose2d(params['nz'] + 128, params['ngf']*8, 
                                           kernel_size=4, stride=1, 
                                           padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x, y):
        
        y = F.leaky_relu(self.fc_embed1(y.squeeze()), 0.2, True)

        x = torch.cat((x, y.view(-1, 128, 1, 1)), dim=1)

        x = F.leaky_relu(self.bn1(self.tconv1(x)), 0.2, True)

        x = F.leaky_relu(self.bn2(self.tconv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.tconv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.tconv4(x)), 0.2, True)

        x = F.tanh(self.tconv5(x))

        return x

# Define the Discriminator Network
class Discriminator_(nn.Module):
    def __init__(self, params):
        super(Discriminator_, self).__init__()

        self.params = params

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimensions: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf']*8, params['ndf']*8, 4, 1, 0, bias=False)
        
        self.linear = nn.Linear(params['ndf']*8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.conv5(x), 0.2, True)
        x = x.view(-1,self.params['ndf']*8) 
        x = self.linear(x)
        x = F.sigmoid(x)
        return x


#wgan
class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.params = params
        self.name = 'Generator'
        self.linear1 = nn.Linear(params['nz'] + params['lsize'], 4*4*8*params['ngf'])
        self.conv1 = nn.ConvTranspose2d(8*params['ngf'], 4*params['ngf'], 2, stride=2)
        self.conv2 = nn.ConvTranspose2d(4*params['ngf'], 2*params['ngf'], 2, stride=2)
        self.conv3 = nn.ConvTranspose2d(2*params['ngf'], params['ngf'], 2, stride=2)
        self.conv4 = nn.ConvTranspose2d(params['ngf'], 3, 2, stride=2)
        self.bn0 = nn.BatchNorm1d(4*4*8*params['ngf'])
        self.bn1 = nn.BatchNorm2d(4*params['ngf'])
        self.bn2 = nn.BatchNorm2d(2*params['ngf'])
        self.bn3 = nn.BatchNorm2d(params['ngf'])
        self.relu = nn.ELU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.bn0(self.linear1(x)))
        x = x.view(-1, 8*self.params['ngf'], 4, 4)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.tanh(x)
        x = x.view(-1, 3, 64, 64)
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.params = params
        self.name = 'Discriminator'
        self.conv1 = nn.Conv2d(3, params['ndf'], 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(params['ndf'], 2*params['ndf'], 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2*params['ndf'], 4*params['ndf'], 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4*params['ndf'], 8*params['ndf'], 3, stride=2, padding=1)
        self.relu = nn.ELU(inplace=True)
        self.linear1 = nn.Linear(4*4*8*params['ndf'], 1)

    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 4*4*8*self.params['ndf'])
        x = self.linear1(x)
        return x


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),  
        nn.ReLU(),
        nn.Dropout(0.5)
    )
    return layer

class VGG16(nn.Module):
    def __init__(self, n_classes=6):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(4*4*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = out
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out

class Classifier(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),  # kernel_size
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x
