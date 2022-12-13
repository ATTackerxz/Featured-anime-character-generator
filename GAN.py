import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.autograd as autograd
import wandb

import os as os
import logging

from dataloader import TrainDataset, dataLoader
from model import weights_init, Classifier, Generator, Discriminator, VGG16



dir_checkpoint = './GAN_checkpoints/'
dir_img = './data_temp'
dir_label = './label'
params = {
    "bsize" : 61,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nepochs' : 100000,# Number of training epochs.
    'lr' : 1e-4,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 10,# Save step.
    'n_critic' : 5, # Number of iterations to train discriminator before training generator.
    'lsize' : 18, # encoded label size
    'ntrain': 11590, # num of training samples
    'nz': 100, # length of noise z
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'l': 10, # lambda of gp
    'mouth_open': 'mouth_open.pt', # load param of mouth_open
    'gender': 'gender.pt',
    'glasses': 'glasses.pt',
    'eyes': 'eyes.pt',
    'hair': 'hair.pt'
    }

def grad_penalty_3dim(params, netD, data, fake):
    alpha = torch.randn(params['bsize'], 1, requires_grad=True).cuda()
    alpha = alpha.expand(params['bsize'], data.nelement()//params['bsize'])
    alpha = alpha.contiguous().view(params['bsize'], 3, 64, 64)
    interpolates = alpha * data + ((1 - alpha) * fake).cuda()
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * params['l']
    return gradient_penalty


def generate_image(params, epoch, label, netG, i):
    with torch.no_grad():
        noise = torch.randn(params['bsize'], params['nz']).cuda()
        imgs = netG(noise, label)

    img = imgs[i]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1).cuda()
    std = torch.tensor([0.5,  0.5, 0.5]).view(3,1,1).cuda()
    img = img.mul_(std).add_(mean)
    _ = wandb.Image(img, caption = "samples_{}".format(epoch))

    path = 'results/samples_{}.png'.format(epoch)
    if not os.path.exists('results'):
        os.makedirs('results')
    print ('saving sample: ', path)
    save_image(img, path)

def train():
    train_dataset = TrainDataset()
    dataset = dataLoader(train_dataset)
    loader_args = dict(batch_size=params['bsize'], num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    train_it = iter(train_loader)

    device = torch.device('cuda')
    eyes = VGG16(6)
    eyes = eyes.to(device)
    eyes.load_state_dict(torch.load(params['eyes'], map_location=device))
    hair = VGG16(6)
    hair = hair.to(device)
    hair.load_state_dict(torch.load(params['hair'], map_location=device))
    mouth_open = Classifier(2)
    mouth_open = mouth_open.to(device)
    mouth_open.load_state_dict(torch.load(params['mouth_open'], map_location=device))
    gender = Classifier(2)
    gender = gender.to(device)
    gender.load_state_dict(torch.load(params['gender'], map_location=device))
    glasses = Classifier(2)
    glasses = glasses.to(device)
    glasses.load_state_dict(torch.load(params['glasses'], map_location=device))

    netG = Generator(params)
    netG = netG.to(device)
    #netG.apply(weights_init)
    netD = Discriminator(params)
    netD = netD.to(device)
    #netD.apply(weights_init)
    optimD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999), weight_decay=1e-4)
    optimG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999), weight_decay=1e-4)

    if not os.path.exists('results/'): 
            os.makedirs('results')

    one = torch.tensor(1.).cuda()
    mone = (one * -1)

    # logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info(f'Using device {device}')

    experiment = wandb.init(project='VE4880J', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=params["nepochs"], batch_size=params["bsize"], learning_rate=params["lr"],
                                    save_checkpoint=True))
    logging.info(f'''Starting training:
            Epochs:          {params["nepochs"]}
            Batch size:      {params["bsize"]}
            Learning rate:   {params["lr"]}
            Training size:   {params["ntrain"]}
            Checkpoints:     {True}
            Device:          {device.type}
        ''')
    global_step = 0

    #training
    for epoch in range(1, params['nepochs'] + 1):
        netD.zero_grad()
        netG.zero_grad()

        
        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(params['n_critic']):
            try:
                batch = next(train_it)
            except:
                train_it = iter(train_loader)
                batch = next(train_it)
            img, label = batch['img'], batch['label']
            img = img.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            netD.zero_grad()
            d_real = netD(img).mean()
            d_real.backward(mone, retain_graph=True)
            noise = torch.randn(params['bsize'], params['nz'], requires_grad=True).cuda()
            with torch.no_grad():
                fake = netG(noise, label)
            fake.requires_grad_(True)
            d_fake = netD(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one, retain_graph=True)
            gp = grad_penalty_3dim(params, netD, img, fake)
            gp.backward()
            d_cost = d_fake - d_real + gp
            wasserstein_d = d_real - d_fake
            optimD.step()

        for p in netD.parameters():
            p.requires_grad=False
        netG.zero_grad()
        try:
            batch = next(train_it)
        except:
            train_it = iter(train_loader)
            batch = next(train_it)
        mouth_open_label, gender_label, glasses_label, eyes_label, hair_label, label = batch['mouth_open'], batch['gender'], batch['glasses'], batch['eyes'], batch['hair'], batch['label']
        noise = torch.randn(params['bsize'], params['nz'], requires_grad=True).cuda()
        label = label.to(device=device, dtype=torch.float32)
        eyes_label = eyes_label.to(device=device, dtype=torch.long)
        hair_label = hair_label.to(device=device, dtype=torch.long)
        mouth_open_label = mouth_open_label.to(device=device, dtype=torch.long)
        gender_label = gender_label.to(device=device, dtype=torch.long)
        glasses_label = glasses_label.to(device=device, dtype=torch.long)
        fake = netG(noise, label)
        # loss with gt

        criterion = nn.CrossEntropyLoss()
        loss_1 = criterion(mouth_open(fake), mouth_open_label.long())
        loss_1.backward(retain_graph=True)
        loss_2 = criterion(gender(fake), gender_label)
        loss_2.backward(retain_graph=True)
        loss_3 = criterion(glasses(fake), glasses_label)
        loss_3.backward(retain_graph=True)
        loss_4 = criterion(eyes(fake), eyes_label)
        loss_4.backward(retain_graph=True)
        loss_5 = criterion(hair(fake), hair_label)
        loss_5.backward(retain_graph=True)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        g_cost = -G + loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        optimG.step()
    
        if epoch % 500 == 0:
            logging.info('epoch{}, train D cost: {}'.format(epoch, d_cost.cpu().item()))
            logging.info('epoch{}, train G cost: {}'.format(epoch, g_cost.cpu().item()))
            generate_image(params, epoch, label, netG, 0)
        if epoch % 5000 == 0:
            torch.save(netG.state_dict(), str(dir_checkpoint + '/G_checkpoint_epoch{}.pth'.format(epoch)))
            torch.save(netD.state_dict(), str(dir_checkpoint + '/D_checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint saved!')

        experiment.log({
            'learning rate D': optimD.param_groups[0]['lr'],
            'learning rate G': optimG.param_groups[0]['lr'],
            'train D cost': d_cost,
            'train G cost': g_cost,
            'step': global_step,
            'epoch': epoch
        })

    

if __name__ == '__main__':
    train()