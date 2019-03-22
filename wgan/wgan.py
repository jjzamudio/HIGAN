
from __future__ import print_function
import argparse
import numpy as np

#import torch.multiprocessing as mp
#mp.set_start_method('spawn')

import torch.nn as nn
import torch

from torch.autograd import Variable
import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

#import h5py
from collections import defaultdict
import pandas as pd
from models import DCGAN_G, DCGAN_D, weights_init
from wgan_utils import check_coords, power_spectrum_np, _gradient_penalty,  HydrogenDataset, get_samples, data_transform
from plot_utils import plot_loss, visualize_cube,  histogram_mean_confint, visualize2d, plot_power_spec

mean_5=14592.24
std_5=922711.56
max_5=4376932000

mean_l5 = 2.7784111
std_l5 = 1.5777067
max_l5 =22.199614


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', required=True, help='path to dataset')
    parser.add_argument('--n_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('--s_sample', type=int, default=64, help='Size of samples')
    parser.add_argument('--workers', type=int, deafult=0, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--epoch_st', type=int, default=0, help='Number of epoch to start')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--lrD', type=float, default=0.0005, help='learning rate for Critic, default=0.0005')
    parser.add_argument('--lrG', type=float, default=0.0005, help='learning rate for Generator, default=0.0005')
    parser.add_argument('--lambda_', type=float, default=10, help='Parameter for Gradient penalty')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--lrdecay',  action='store_true', help='Learning rate decay')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--load_weights', action='store_true', help='Load saved weights' )
    parser.add_argument('--load_opt', action='store_true', help='Load saved optimizers')
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--fixed', action='store_true', help='Use fixed samples' )
    opt = parser.parse_args()
    print(opt)
   
     
    n_samples = int(opt.n_samples)
    s_sample = int(opt.s_sample)
    batchSize = int(opt.batchSize)
    ngpu = int(opt.ngpu)
    n_extra_layers = int(opt.n_extra_layers)
    
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 1           #By default our data has only one channel
    niter = int(opt.niter)
    Diters = int(opt.Diters)
    
    workers = int(opt.workers)
    lambda_ = int(opt.lambda_)
    cuda = opt.cuda
    
    #datapath='../../../../../'
    #f = h5py.File(opt.datapath+'fields_z='+opt.redshift+'.hdf5', 'r')
    #f = f['delta_HI']  
    
    netG = DCGAN_G(s_sample, nz, nc, ngf, ngpu, n_extra_layers)
    netD = DCGAN_D(s_sample, nz, nc, ndf, ngpu, n_extra_layers)
    
    #experiments/ch128_lr0005_tanh/netD_epoch_47.pth
    epoch_load = opt.epoch_st - 1
    wass_loss=[]
        
    if opt.load_weights == True:
        netG.load_state_dict(torch.load(opt.experiment+'netG_epoch_' + str(epoch_load) + '.pth'))
        netD.load_state_dict(torch.load(opt.experiment+'netD_epoch_' + str(epoch_load) + '.pth'))
        
        wass_loss=pd.read_csv(opt.experiment +'loss.csv', header=None)

        wass_loss=wass_loss[wass_loss.columns[0]].tolist()
        
    device = torch.device("cuda" if opt.cuda else "cpu")
    
     #part, datapath, s_sample, nsamples, transform, d2

    partition = {'train': [x for x in range(0, n_samples)]}
    
    dataset = HydrogenDataset(part = partition,
                              datapath=opt.datapath,
                                s_sample = s_sample, 
                                transform='log_max',
                                d2=False,
                                mode = opt.fixed)
    
    shuff = True
    if opt.fixed == False:
        shuff = False

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size = batchSize,
                                             shuffle = shuff, 
                                             num_workers=int(opt.workers),
                                            drop_last = True)
    
    print(netG)
    print(netD)
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    errG_l=[]
    errD_real_l=[]
    errD_fake_l=[]
    
    input = torch.FloatTensor(batchSize, 1, s_sample, s_sample, s_sample)
    noise = torch.FloatTensor(batchSize, nz,1 , 1, 1,  device=device).normal_(0,1)
    fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1, 1 ).normal_(0, 1)
    
    one = torch.FloatTensor([1])
    #one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    
    #torch.cuda.empty_cache()
    if cuda==True:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
    
    #experiments/ch256_lr0005/optG_41.pth
    
    if opt.load_opt == True:
        optimizerD.load_state_dict(torch.load(opt.experiment+'optD_'+str(epoch_load)+'.pth'))
        optimizerG.load_state_dict(torch.load(opt.experiment+'optG_'+str(epoch_load)+'.pth'))
        
        optimizerD.state = defaultdict(dict, optimizerD.state)
        optimizerG.state = defaultdict(dict, optimizerG.state)
        #print('opts loaded')
    
    gen_iterations = 0
    for epoch in range(opt.epoch_st, niter + opt.epoch_st):
        #print (epoch)
         #Learning rate decay
        if opt.lrdecay and (epoch+1) == 5:
            optimizerD.param_groups[0]['lr'] /= 10
            optimizerG.param_groups[0]['lr'] /= 10
            print("Learning rate change")
        
        if opt.lrdecay and (epoch+1) == 10:
            optimizerD.param_groups[0]['lr'] /= 10
            optimizerG.param_groups[0]['lr'] /= 10
            print("2nd learning rate change ")
            
              
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
               
            if gen_iterations < 5 or gen_iterations % 500 == 0:
                Diters = 1
            else:
                Diters = opt.Diters
                
            j=0
            while j < Diters and i < len(dataloader):
                j += 1
                    
                data=data_iter.next()
                i+=1
                
                real_cpu= data
                netD.zero_grad()
              
                batch_size=real_cpu.size(0)
                if cuda==True:
                    real_cpu=real_cpu.cuda()
            
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)   
                errD_real=netD(inputv)
                
                errD_real.backward(one)
                    
                #Train with fake
                noise.resize_(batchSize, nz, 1, 1, 1).normal_(0, 1)
    
                with torch.no_grad():
                    noisev = Variable(noise) # totally freeze netG
                    
                fake = Variable(netG(noisev).data)
    
                inputv=fake
                   
                errD_fake = netD(inputv)
                errD_fake.backward(mone)
                
                ## train with gradient penalty
                gradient_penalty = _gradient_penalty(netD, real_cpu.data, fake.data, lambda_)
                gradient_penalty.backward()
                errD = errD_real - errD_fake + gradient_penalty
                wass_D =  errD_real - errD_fake
                
                optimizerD.step()
                wass_loss.append(float(wass_D.data[0]))
                    
                errD_real_l.append(float(errD_real.data[0]))
                errD_fake_l.append(float(errD_fake.data[0]))
               
               
            ############################
            # (2) Update G network
            ###########################
    
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batchSize, nz, 1, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            #print('G:',fake.shape)
            errG = netD(fake)
            
            #print('errG: ', errG)
            #label = torch.full(size = (batch_size,), fill_value = real_label, device = device)
            errG.backward(one)
            optimizerG.step()
            
            gen_iterations += 1
            
            if lambda_ is not None:
                errD=wass_D
            
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, niter+ opt.epoch_st, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            
            #wass_loss.append(float(errD.data[0]))
            errG_l.append(float(errG.data[0]))
    
            if gen_iterations % 1 == 0:
                with torch.no_grad():
                    fake = netG(Variable(fixed_noise))
                    fake = np.array(fake)
                    fake_i = data_transform(fake, 'log_max', inverse=True)
                    
                    real_cpu = np.array(real_cpu)
                    
    
                plot_power_spec(fake_i, mean_5, s_size=s_sample, log_scale=True, BoxSize=(75.0/2048.0)*s_sample,
                               save_plot=opt.experiment, t=gen_iterations)
                
                visualize2d(real_cpu, fake, log= False, save=opt.experiment, t=gen_iterations, show_plot=False)
                
                histogram_mean_confint(fake, real_cpu, log_plot=False, t = gen_iterations, 
                                       save_plot=opt.experiment, show_plot=False)

   
                plot_loss(wass_loss,'Wasserstein loss', log_=True, save_plot=opt.experiment, t=gen_iterations, med_filter=True)
   
                
                
                torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
                torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
                torch.save(optimizerD.state_dict(), '{0}/optD_{1}.pth'.format(opt.experiment, epoch))
                torch.save(optimizerG.state_dict(), '{0}/optG_{1}.pth'.format(opt.experiment, epoch))
                
                loss = pd.DataFrame(wass_loss)
                loss.to_csv(opt.experiment+'loss.csv', index=False, header=False)
               
        # do checkpointing
        loss = pd.DataFrame(wass_loss)
        loss.to_csv(opt.experiment+'loss.csv', index=False, header=False)
        
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(optimizerD.state_dict(), '{0}/optD_{1}.pth'.format(opt.experiment, epoch))
        torch.save(optimizerG.state_dict(), '{0}/optG_{1}.pth'.format(opt.experiment, epoch))

