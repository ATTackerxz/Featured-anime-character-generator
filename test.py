import torch
from torchvision.utils import save_image
import numpy as np
import os as os
import logging

from dataloader import TrainDataset, dataLoader
from model import weights_init, Classifier, Generator, Discriminator



dir_checkpoint = './GAN_checkpoints/'
dir_img = './data_temp'
dir_label = './label'
params = {
    "bsize" : 18,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nepochs' : 10000,# Number of training epochs.
    'lr' : 1e-4,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 10,# Save step.
    'n_critic' : 5, # Number of iterations to train discriminator before training generator.
    'lsize' : 18, # encoded label size
    'ntrain': 1921, # num of training samples
    'nz': 100, # length of noise z
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'l': 10, # lambda of gp
    'mouth_open': 'mouth_open.pt', # load param of mouth_open
    'gender': 'gender.pt',
    'glasses': 'glasses.pt',
    'G': './GAN_checkpoints/G_checkpoint_epoch45000.pth'
    }

def generate_image(params, label, netG):
    with torch.no_grad():
        noise = torch.randn(params['bsize'], params['nz']).cuda()
        img = netG(noise, label)

    for i in range(len(img)):
        path = 't_results/samples_{}.png'.format(i)
        if not os.path.exists('results'):
            os.makedirs('results')
        print ('saving sample: ', path)
        save_image(img[i], path)

def test():
    
    device = torch.device('cuda')

    netG = Generator(params)
    netG = netG.to(device)
    netG.load_state_dict(torch.load(params['G'], map_location=device))
    
    if not os.path.exists('t_results/'): 
            os.makedirs('t_results')

    # logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info(f'Using device {device}')


    logging.info(f'''Starting test:
            
        ''')

    label = torch.from_numpy(np.array([[1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0],
                                        [1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0],
                                        [1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0],
                                        [1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1]
                                        ])).to(torch.long) 
    label = label.to(device=device, dtype=torch.float32)
    generate_image(params, label, netG)
        

if __name__ == '__main__':
    test()