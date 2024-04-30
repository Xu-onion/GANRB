import os
import pandas as pd
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ganrb import Discriminator, Generator
from preprocessing import Dataset
import datetime
import warnings
warnings.filterwarnings('ignore')

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=str, default='0')
parser.add_argument('--is_train', default=True, action='store_true')
parser.add_argument("--n_epochs", type=int, default=5000, help="number of training epochs")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--data_len', type=int, default=70, help="length of spectrum")
parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_id
#############################################################################################################################################

# load data function
class raman_dataset(Dataset):
    def __init__(self, file_path, raman_file, cars_file):
        self.raman_data = pd.read_csv(os.path.join(file_path, raman_file)).iloc[:, 1:]
        self.cars_data = pd.read_csv(os.path.join(file_path, cars_file)).iloc[:, 1:]

    def __len__(self):
        return len(self.raman_data)

    def __getitem__(self, idx):
        raman_data = self.raman_data.values[idx]
        cars_data = self.cars_data.values[idx]
        return raman_data, cars_data

D_loss = []
G_loss = []
avg_mae = []
avg_mse = []
# Build a DataFrame to save history
history = pd.DataFrame()
if opt.is_train:
    # load training data
    dataset_train = raman_dataset('data', '1aRaman_spectrums_train.csv', '1aCARS_spectrums_train.csv')
    trainloader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

    dataset_val = raman_dataset('data', '1aRaman_spectrums_valid.csv', '1aCARS_spectrums_valid.csv')
    val_loader = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False)

    # init netD and netG
    netD = Discriminator(data_len=opt.data_len).cuda()
    netG = Generator(data_len=opt.data_len).cuda()

    criterion = nn.BCELoss().cuda()
    l1_loss = nn.L1Loss().cuda()        # MAE
    l2_loss = nn.MSELoss().cuda()       # MSE
    sml1_loss = nn.SmoothL1Loss().cuda()

    ## Adam
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    start_time = datetime.datetime.now()
    for epoch in range(opt.n_epochs):
        print("Epoch:", epoch)
        # train------
        D_epoch_loss = 0
        G_epoch_loss = 0
        count = len(trainloader)
        for step, (real_signal, input_signal) in enumerate(trainloader):
            netD.train()
            netG.train()
            input_signal = input_signal.float().cuda()
            real_signal = real_signal.float().cuda()
            # generate fake signal
            fake_signal = netG(input_signal)

            ######################
            # (1) Update G network
            ######################
            optimizerG.zero_grad()

            # First, G(A) should fake the discriminator
            disc_fake_out = netD(fake_signal, input_signal)
            gen_loss_crossentropyloss = criterion(disc_fake_out, torch.ones_like(disc_fake_out).cuda())

            # Second, G(A) = B
            # gen_l1_loss = l1_loss(fake_signal, input_signal)                 # L1 LOSS
            # gen_l2_loss = l2_loss(fake_signal, input_signal)                 # L2 loss
            gen_sml1_loss = sml1_loss(fake_signal, input_signal)               # Smooth L1 loss
            # G_loss = BCE + Smooth L1
            gen_loss = gen_loss_crossentropyloss + 100 * gen_sml1_loss

            gen_loss.backward()
            optimizerG.step()

            ######################
            # (2) Update D network
            ######################
            # if epoch % 5 == 0:
            optimizerD.zero_grad()

            # train with real
            disc_real_output = netD(real_signal, input_signal)
            d_real_loss = criterion(disc_real_output, torch.ones_like(disc_real_output).cuda())

            # train with fake
            disc_fake_output = netD(fake_signal.detach(), input_signal)
            d_fake_loss = criterion(disc_fake_output, torch.zeros_like(disc_fake_output).cuda())

            # Combined D loss
            disc_loss = (d_real_loss + d_fake_loss) * 0.5
            disc_loss.backward()
            optimizerD.step()

            with torch.no_grad():
                D_epoch_loss += disc_loss.item()
                G_epoch_loss += gen_loss.item()

            # print('TRAIN:[%d/%d][%d/%d]\td_fake_loss: %.6f\td_real_loss: %.6f\tLoss_D: %.6f\tLoss_G: %.6f'
            #       % (epoch, epoch_num, step, count, d_fake_loss, d_real_loss, disc_loss, gen_loss))

        # Validate------
        netD.eval()
        netG.eval()
        with torch.no_grad():
            D_epoch_loss /= count
            G_epoch_loss /= count
            D_loss.append(D_epoch_loss)
            G_loss.append(G_epoch_loss)
            elapsed_time = datetime.datetime.now() - start_time
            print('TRAIN:[%d/%d][%d/%d]\tLoss_D: %.6f\tLoss_G: %.6f\t time: %s'
                  % (epoch, opt.n_epochs, step, len(trainloader), D_loss[-1], G_loss[-1], elapsed_time))

            all_mse = 0
            all_mae = 0
            count = len(val_loader)
            for step, (real_signal_val, input_signal_val) in enumerate(val_loader):
                real_signal_val = real_signal_val.float().cuda()
                input_signal_val = input_signal_val.float().cuda()

                prediction = netG(input_signal_val)
                mae = l1_loss(prediction, real_signal_val)
                mse = l2_loss(prediction, real_signal_val)
                all_mae += mae
                all_mse += mse

                # elapsed_time = datetime.datetime.now() - start_time
                # print('VAL:[%d/%d][%d/%d]\tMSE: %.4f\t time: %s'
                #       % (epoch, epoch_num, step, len(val_loader), mse, elapsed_time))

            avg_mae.append(all_mae / count)
            avg_mse.append(all_mse / count)
            elapsed_time = datetime.datetime.now() - start_time
            print('VAL:[%d/%d][%d/%d]\tMAE: %.6f\t    MSE: %.6f\t     time: %s'
                  % (epoch, opt.n_epochs, step, count, avg_mae[-1], avg_mse[-1], elapsed_time))

            # Visualize images in training
            # if epoch % 10 == 0:
            if not os.path.exists("img"):
                os.mkdir("img")
            real_signal_batch, input_signal_batch = next(iter(val_loader))
            input_signal_batch = input_signal_batch.float().cuda()
            prediction = netG(input_signal_batch).detach().cpu().numpy()
            test_anno = real_signal_batch.cpu().numpy()
            test_data = input_signal_batch.cpu().numpy()
            titles = ['Input', 'Ground Truth', 'Output']
            display_list = [test_data, test_anno, prediction]

            fig, axs = plt.subplots(8, 3, figsize=(16, 20))
            for i in range(8):
                for j in range(3):
                    axs[i, j].plot(display_list[j][i])
                    axs[i, j].set_title(titles[j])
                    axs[i, j].axis('off')
            plt.savefig('./img/%d.png' % epoch)
            plt.close()

        # save trained model
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if epoch % 1 == 0:
            netG_out_path = "./checkpoint/GANRB_G%s.pth" % epoch
            torch.save(netG.state_dict(), netG_out_path)
            # netD_out_path = "./nets/GANRB_D%s.pth" % epoch
            # torch.save(netD.state_dict(), netD_out_path)

        # save history
        epoch_result = dict(epoch=epoch,
                            D_loss=round(float(D_loss[-1]), 6), G_loss=round(float(G_loss[-1]), 6),
                            avg_mae=round(float(avg_mae[-1]), 6), avg_mse=round(float(avg_mse[-1]), 6),
                            )
        history = history._append(epoch_result, ignore_index=True)
        history.to_csv("./checkpoint/history.csv", index=False)

    # Plot loss
    x_ = range(0, opt.n_epochs)
    plt.plot(x_, D_loss, label='Discriminator Losses')
    plt.plot(x_, np.array(G_loss) / 100, label='Generator Losses / 100')
    plt.legend()
    plt.savefig('./checkpoint/loss.png')
    plt.close()
