import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

from utils import *
from TO_msnet import MSNet as TO_MSNet
from ftanet_harm import FTAnet
import time
from tqdm import tqdm

import argparse


class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_tensor=None):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        if self.target_tensor is not None:
            return self.data_tensor[index], self.target_tensor[index]
        else:
            return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def train(train, test, epoch_num, batch_size, lr, gid, op, semi, pretrained_to=None, pretrained_ms=None):
    print('batch size:', batch_size)
    print('epoch_num', epoch_num)
    torch.backends.cudnn.enabled = False
    gid = list(map(int, gid.split(",")))
    device = torch.device("cuda:{}".format(gid[0]))
    ftanet = FTAnet()
    tonet = TO_MSNet()
    tonet = torch.nn.DataParallel(tonet, device_ids=gid)
    ftanet = torch.nn.DataParallel(ftanet, device_ids=gid)
    print('gid:', gid)
    if pretrained_to is not None:
        tonet.load_state_dict(torch.load(pretrained_to, weights_only=True), strict=True)

    if pretrained_ms is not None:
        ftanet.load_state_dict(torch.load(pretrained_ms, weights_only=True), strict=True)

    if gid is not None:
        ftanet.to(device=device)
        tonet.to(device=device)
    else:
        ftanet.cpu()
    ftanet.float()
    tonet.float()

    epoch_num = epoch_num
    lr = lr

    X_train, y_train = load_train_data(path=train)
    y_train = build_harmonic3(f02img(y_train))

    test_list = load_list(path=test, mode='test')
    X_target = load_semi_data(path=semi)

    data_set = Dataset(data_tensor=X_train, target_tensor=y_train)
    semi_data_set = Dataset(data_tensor=X_target)
    semi_data_loader = Data.DataLoader(dataset=semi_data_set, batch_size=batch_size, shuffle=True, drop_last =True)

    best_epoch = 0
    best_OA = 0

    BCELoss = nn.BCEWithLogitsLoss()
    CrossLoss = nn.CrossEntropyLoss()
    opt = optim.Adam(ftanet.parameters(), lr=lr)
    tick = time.time()
    for epoch in range(epoch_num):
        list_x=[]
        list_y=[]
        for step, batch_x in enumerate(tqdm(semi_data_loader)):
            batch_x = batch_x.to(device)
            with torch.no_grad():
                _, pred_oct, pred_tone, feat_oct, feat_tone = tonet(batch_x)
                pred_f0, _ = ftanet(batch_x)
                pred_f0 = pred_f0[:,0,:,:]
                pred_oct = pred_oct.argmax(dim=1)
                pred_tone = pred_tone.argmax(dim=1)
                pred_f0 = pred_f0.argmax(dim=1)
                label_oct = freq2octave(pred_f0)
                label_tone = freq2tone(pred_f0)

                mask0 = pred_f0!=0
                label_total = torch.sum(mask0, dim=1)
                mask = (label_oct == pred_oct) * (label_tone == pred_tone)
                mask = mask * mask0
                total_correct = torch.sum(mask, dim=1)
                mask = total_correct > label_total * 0.8

                if mask.any():
                    list_x.append(batch_x[mask])
                    list_y.append(pred_f0[mask])
        x_tensor = torch.cat(list_x, dim=0).to('cpu')
        y_tensor = torch.cat(list_y, dim=0)
        y_tensor = build_harmonic3(f02img(y_tensor)).to('cpu')
        print('x_tensor:{} y_tensor:{}'.format(x_tensor.shape, y_tensor.shape))
        select_data_set = Dataset(data_tensor=x_tensor, target_tensor=y_tensor)

        data_loader = Data.DataLoader(dataset=data_set + select_data_set, batch_size=batch_size, shuffle=True, drop_last =True)
        if len(select_data_set) == 0:
            print('************** no sample ****************')
            continue
        else:
            print('sample{}/{}'.format(len(select_data_set),len(semi_data_set) ))

        tick_e = time.time()
        ftanet.train()
        train_loss = 0

        for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

            opt.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if gid is not None:
                pred, _ = ftanet(batch_x)
                loss = CrossLoss(pred, batch_y.argmax(dim=1))
            else:
                pred = ftanet(batch_x)
                loss = BCELoss(pred, batch_y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        ftanet.eval()
        eval_arr = np.zeros(5, dtype=np.double)
        with torch.no_grad():
            tick_r = time.time()
            for i in range(len(test_list)):
                pred_list = []
                X_test, y_test = load_data(test_list[i])
                X_test = X_test.to(device)
                split_total = torch.split(X_test, split_size_or_sections=16)
                for i, split_x in enumerate(split_total):
                    pred_list.append(ftanet(split_x)[0])
                pred = torch.cat(pred_list, dim=0)
                est_freq = pred2res(pred[:,0,:,:]).flatten()
                ref_freq = y2res(y_test).flatten()

                time_series = np.arange(len(ref_freq)) * 0.01
                eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)

            print('ref.time', time.time() - tick_r)
            eval_arr /= len(test_list)
            train_loss /= step + 1

        print("Epoch={:3d}\tTrain_loss={:6.4f}\tLearning_rate={:6.4f}e-4".format(epoch, train_loss, 1e4 *
                                                                                 opt.state_dict()['param_groups'][0][
                                                                                     'lr']))
        print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                       eval_arr[2], eval_arr[3],
                                                                                       eval_arr[4]))
        if eval_arr[-1] > best_OA:
            best_OA = eval_arr[-1]
            best_epoch = epoch
        torch.save(ftanet.state_dict(), op + '{:.2f}_{:d}.pt'.format(eval_arr[4], epoch))
        print('Best Epoch: ', best_epoch, ' Best OA: ', best_OA)
        print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))


def parser():
    p = argparse.ArgumentParser()

    p.add_argument('-train', '--train_list_path',
                   help='the path of training data list (default: %(default)s)',
                   type=str, default='./train_dataset.txt')
    p.add_argument('-test', '--test_list_path',
                   help='the path of test data list (default: %(default)s)',
                   type=str, default='./test_04_npy.txt')
    p.add_argument('-semi', '--semi_list_path',
                   help='the path of semi data list (default: %(default)s)',
                   type=str, default='./semi_dataset.txt')
    p.add_argument('-ep', '--epoch_num',
                   help='the number of epoch (default: %(default)s)',
                   type=int, default=100)
    p.add_argument('-bs', '--batch_size',
                   help='The number of batch size (default: %(default)s)',
                   type=int, default=32)
    p.add_argument('-lr', '--learning_rate',
                   help='the number of learn rate (default: %(default)s)',
                   type=float, default=0.0002)
    p.add_argument('-gpu', '--gpu_index',
                   help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s)',
                   type=str, default="0,1")
    p.add_argument('-o', '--output_dir',
                   help='Path to output folder (default: %(default)s)',
                   type=str, default='./model/')
    p.add_argument('-pm', '--pretrained_model',
                   help='the path of pretrained model (Transformer or Streamline) (default: %(default)s)',
                   type=str)

    return p.parse_args()


if __name__ == '__main__':
    args = parser()
    gid = args.gpu_index
    gid = list(map(int, gid.split(",")))[0]
    pretrained_ftanet = 'pretrained_ftanet.pth'
    pretrained_tonet = 'pretrained_tonet.pth'

    if args.gpu_index is not None:
        train(args.train_list_path, args.test_list_path, args.epoch_num, args.batch_size, args.learning_rate,
              args.gpu_index, args.output_dir, args.semi_list_path ,pretrained_tonet, pretrained_ftanet)
    else:
        train(args.train_list_path, args.test_list_path, args.epoch_num, args.batch_size, args.learning_rate,
              args.gpu_index, args.output_dir, args.pretrained_model)
