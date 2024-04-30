import os
import pandas as pd
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from preprocessing import Dataset
import datetime
import warnings
warnings.filterwarnings('ignore')

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=str, default='0')
parser.add_argument('--is_test', default=True, action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--data_len', type=int, default=70, help="length of spectrum")
parser.add_argument('--model_name', type=str, default='GANRB_G', help="name of model")

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


if opt.is_test:
    start_time = datetime.datetime.now()
    # load data
    dataset_val = raman_dataset('data', '1aRaman_spectrums_valid.csv', '1aCARS_spectrums_valid.csv')
    val_loader = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False)

    # load model
    model_path = './checkpoint/%s.pth' % opt.model_name
    netG = torch.load(model_path)

    if not os.path.exists("pred/"):
        os.makedirs("pred/")

    netG.cuda()
    with torch.no_grad():
        [w, h] = dataset_val.cars_data.shape
        results = np.empty([0, h])
        for step, (real_signal_val, input_signal_val) in enumerate(val_loader):
            real_signal_val = real_signal_val.float().cuda()
            input_signal_val = input_signal_val.float().cuda()

            prediction = netG(input_signal_val).detach().cpu().numpy()
            results = np.append(results, prediction, axis=0)

            test_anno = real_signal_val.cpu().numpy()
            test_data = input_signal_val.cpu().numpy()

            x_ = np.arange(0, opt.data_len)
            font = {'family': 'Times New Roman','weight': 'normal','size': 24}
            plt.figure(figsize=(20, 10), dpi=100)
            my_x_ticks = np.arange(0, opt.data_len, 1)
            plt.xticks(my_x_ticks)
            plt.xlim((0, 70))
            # plt.ylim((0, 1))
            plt.plot(x_, test_data[0], 'b', linewidth=3, label='Cars')
            plt.plot(x_, test_anno[0], 'g', linewidth=3, label='Raman')
            plt.plot(x_, prediction[0], 'r', linewidth=3, label='Prediction')
            plt.xlabel('Wavenumber', font)
            plt.ylabel('Intensity', font)
            plt.legend(fontsize=24)
            # plt.show()
            plt.savefig('./pred/%d.png' % step)
            plt.close()

        elapsed_time = datetime.datetime.now() - start_time
        print('Inference---- time: %s' % (elapsed_time))

    pd.DataFrame(results).to_csv('./pred/%s.csv' % opt.model_name)











