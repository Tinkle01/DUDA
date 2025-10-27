import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import random
from utils import *
from TO_msnet import MSNet
import time
import argparse

class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_bin, expand = None):
        self.target_bin = target_bin
        self.data_tensor = data_tensor
        octave = freq2octave(target_bin)
        tone = freq2tone(target_bin)
        self.target_octave = octave2img(octave)
        self.target_tone = tone2img(tone)
        self.expand = expand

    def __getitem__(self, index):
        if self.expand:
            index = index % self.data_tensor.size(0)
        return self.data_tensor[index], self.target_octave[index], self.target_tone[index], self.target_bin[index]

    def __len__(self):
        if self.expand:
            return self.expand
        else:
            return self.data_tensor.size(0)

class Dataset_target(Data.Dataset):
    def __init__(self, data_tensor, expand=None):
        self.data_tensor = data_tensor
        self.expand = expand

    def __getitem__(self, index):
        if self.expand:
            index = index % self.data_tensor.size(0)
        index2 = random.randint(0, self.data_tensor.size(0)-1)
        index3 = random.randint(0, self.data_tensor.size(0)-1)

        return self.data_tensor[index], self.data_tensor[index2], self.data_tensor[index3]

    def __len__(self):
        if self.expand:
            return self.expand
        else:
            return self.data_tensor.size(0)


def train(train, test, epoch_num, batch_size, lr, gid, op, semi, pretrained_t=None, pretrained_s=None):
    print('batch size:', batch_size)
    torch.backends.cudnn.enabled = False

    gid = list(map(int, gid.split(",")))

    device = torch.device("cuda:1")

    TeacherA = MSNet()
    TeacherB = MSNet()
    StudentA = MSNet()
    StudentB = MSNet()

    TeacherA = torch.nn.DataParallel(TeacherA, device_ids=gid)
    TeacherB = torch.nn.DataParallel(TeacherB, device_ids=gid)
    StudentA = torch.nn.DataParallel(StudentA, device_ids=gid)
    StudentB = torch.nn.DataParallel(StudentB, device_ids=gid)

    if pretrained_t is not None:
        TeacherA.load_state_dict(torch.load(pretrained_t, weights_only=True), strict=True)
        TeacherB.load_state_dict(torch.load(pretrained_t, weights_only=True), strict=True)
        StudentA.load_state_dict(torch.load(pretrained_s, weights_only=True), strict=True)
        StudentB.load_state_dict(torch.load(pretrained_s, weights_only=True), strict=True)

    if gid is not None:
        TeacherA.to(device=device)
        TeacherB.to(device=device)
        StudentA.to(device=device)
        StudentB.to(device=device)
    else:
        TeacherA.cpu()
        TeacherB.cpu()
        StudentA.cpu()
        StudentB.cpu()

    TeacherA.float()
    TeacherB.float()
    StudentA.float()
    StudentB.float()

    epoch_num = epoch_num
    lr = lr

    X_train, y_train = load_train_data(path=train)
    X_target = load_semi_data(path=semi)

    data_set = Dataset(data_tensor=X_train, target_bin=y_train.cpu())
    data_loader = Data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, drop_last=True)

    target_dataset = Dataset_target(data_tensor=X_target, expand=len(data_set))
    target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_list = load_list(path=test, mode='test')


    best_epoch_B = 0
    best_epoch_A = 0
    best_OA_B = 0
    best_OA_A = 0
    time_series = np.arange(64) * 0.01

    BCELoss = nn.BCEWithLogitsLoss()
    KLLoss = nn.KLDivLoss()
    CrossLoss = nn.CrossEntropyLoss()
    opt = optim.Adam(StudentA.parameters(), lr=lr)
    opt2 = optim.Adam(StudentB.parameters(), lr=lr)

    tick = time.time()

    StudentA.train()
    StudentB.train()
    for epoch in range(epoch_num):

        mask_ST_1 = torch.randint(0, 2, (1, 64))
        mask_ST_1 = mask_ST_1.unsqueeze(dim=1)
        maskX_ST_1 = mask_ST_1.unsqueeze(dim=1).repeat(batch_size, 3, 320, 1)
        maskY_ST_1_oct = mask_ST_1.repeat(batch_size, 7, 1).to(device)
        maskY_ST_1_tone = mask_ST_1.repeat(batch_size, 13, 1).to(device)

        mask_ST_2 = torch.randint(0, 2, (1, 64))
        mask_ST_2 = mask_ST_2.unsqueeze(dim=1)
        maskX_ST_2 = mask_ST_2.unsqueeze(dim=1).repeat(batch_size, 3, 320, 1)
        maskY_ST_2_oct = mask_ST_2.repeat(batch_size, 7, 1).to(device)
        maskY_ST_2_tone = mask_ST_2.repeat(batch_size, 13, 1).to(device)

        mask_ST_3 = torch.randint(0, 2, (1, 64))
        mask_ST_3 = mask_ST_3.unsqueeze(dim=1)
        maskX_ST_3 = mask_ST_3.unsqueeze(dim=1).repeat(batch_size, 3, 320, 1)
        maskY_ST_3_oct = mask_ST_3.repeat(batch_size, 7, 1).to(device)
        maskY_ST_3_tone = mask_ST_3.repeat(batch_size, 13, 1).to(device)

        mask_TT = torch.randint(0, 2, (1, 64))
        mask_TT = mask_TT.unsqueeze(dim=1)
        maskX_TT = mask_TT.unsqueeze(dim=1).repeat(batch_size, 3, 320, 1)
        maskY_TT_oct = mask_TT.repeat(batch_size, 7, 1).to(device)
        maskY_TT_tone = mask_TT.repeat(batch_size, 13, 1).to(device)


        tick_e = time.time()
        train_loss = 0
        for step, (source, target) in enumerate(zip(data_loader, target_loader)):
            opt.zero_grad()
            batch_x, batch_oct, batch_tone, _ = source
            target1, target2, target3 = target

            ST1 = maskX_ST_1 * batch_x + (1-maskX_ST_1) * target1
            ST2 = maskX_ST_2 * batch_x + (1-maskX_ST_2) * target2

            batch_x = batch_x.to(device)
            batch_oct = batch_oct.to(device)
            batch_tone = batch_tone.to(device)

            with torch.no_grad():
                _, label_T1_oct, label_T1_tone = TeacherA(target1.to(device))
                label_ST1_oct = maskY_ST_1_oct * batch_oct + (1-maskY_ST_1_oct) * label_T1_oct
                label_ST1_tone = maskY_ST_1_tone * batch_tone + (1 - maskY_ST_1_tone) * label_T1_tone

                _, label_T2_oct, label_T2_tone = TeacherB(target2.to(device))
                label_ST2_oct = maskY_ST_2_oct * batch_oct + (1 - maskY_ST_2_oct) * label_T2_oct
                label_ST2_tone = maskY_ST_2_tone * batch_tone + (1 - maskY_ST_2_tone) * label_T2_tone

            _, pred_S_oct, pred_S_tone = StudentA(batch_x)
            _, pred_ST1_oct, pred_ST1_tone = StudentA(ST1)
            _, pred_ST2_oct, pred_ST2_tone = StudentA(ST2)


            loss = (0.3* BCELoss(pred_S_oct, batch_oct) + 0.3* BCELoss(pred_S_tone, batch_tone) +
                    0.1*BCELoss(pred_ST1_oct, label_ST1_oct) + 0.1*BCELoss(pred_ST1_tone, label_ST1_tone)
                    + 0.1 * BCELoss(pred_ST2_oct, label_ST2_oct) + 0.1 * BCELoss(pred_ST2_tone, label_ST2_tone)
                    )

            loss.backward()
            opt.step()
            train_loss += loss.item()

        update_ema_variables(StudentA, TeacherA, 0.9999, epoch + 1)

        StudentA.eval()
        eval_arr = np.zeros(5, dtype=np.double)
        with torch.no_grad():
            for i in range(len(test_list)):
                X_test, y_test = load_data(test_list[i])
                X_test = X_test.to(device)
                pred_list1 = []
                pred_list2 = []
                if gid is not None:
                    split_X = torch.split(X_test, split_size_or_sections=16)
                    for i, split in enumerate(split_X):
                        _, output2, output3 = StudentA(split)
                        pred_list1.append(output2)
                        pred_list2.append(output3)
                    pred_octave = torch.cat(pred_list1,dim=0)
                    pred_tone = torch.cat(pred_list2, dim=0)

                else:
                    pred = StudentA(X_test)
                pred_octave = pred_octave.argmax(dim=1)
                pred_tone = pred_tone.argmax(dim=1)
                est_freq = note2freq(pred_octave, pred_tone).flatten()
                label_octave = freq2octave(y_test)
                label_tone = freq2tone(y_test)
                ref_freq = note2freq(label_octave, label_tone).flatten()
                time_series = np.arange(len(ref_freq)) * 0.01
                eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)

            eval_arr /= len(test_list)

            train_loss /= step + 1


        print("---------Student A------------")
        print("Epoch={:3d}\tTrain_loss={:6.4f}\tLearning_rate={:6.4f}e-4".format(epoch, train_loss, 1e4 *
                                                                                 opt.state_dict()['param_groups'][
                                                                                     0][
                                                                                     'lr']))
        print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                       eval_arr[2], eval_arr[3],
                                                                                       eval_arr[4]))
        if eval_arr[-1] > best_OA_A:
            best_OA_A = eval_arr[-1]
            best_epoch_A = epoch

        torch.save(StudentA.state_dict(), op + '{:.2f}_{:d}'.format(eval_arr[4], epoch))
        print('Best Epoch: ', best_epoch_A, ' Best OA: ', best_OA_A)
        print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))

        tick_e = time.time()
        train_loss = 0
        StudentB.train()
        for step, (source, target) in enumerate(zip(data_loader, target_loader)):
            opt2.zero_grad()
            batch_x, batch_oct, batch_tone, _ = source
            target1, target2, target3 = target

            ST1 = maskX_ST_1 * batch_x + (1-maskX_ST_1) * target1
            ST2 = maskX_ST_2 * batch_x + (1-maskX_ST_2) * target2
            ST3 = maskX_ST_3 * batch_x + (1 - maskX_ST_3) * target3
            TT = maskX_TT * target1 + (1 - maskX_TT) * target2

            batch_x = batch_x.to(device)
            batch_oct = batch_oct.to(device)
            batch_tone = batch_tone.to(device)
            with torch.no_grad():
                _, label_T3_oct, label_T3_tone = TeacherB(target3.to(device))
                label_ST3_oct = maskY_ST_3_oct * batch_oct + (1 - maskY_ST_3_oct) * label_T3_oct
                label_ST3_tone = maskY_ST_3_tone * batch_tone + (1 - maskY_ST_3_tone) * label_T3_tone

                _, label_T1_oct, label_T1_tone = StudentA(target1.to(device))
                _, label_T2_oct, label_T2_tone = StudentA(target2.to(device))
                label_TT_oct = maskY_TT_oct * label_T1_oct + (1 - maskY_TT_oct) * label_T2_oct
                label_TT_tone = maskY_TT_tone * label_T1_tone + (1 - maskY_TT_tone) * label_T2_tone

                label_ST1_oct = maskY_ST_1_oct * batch_oct + (1 - maskY_ST_1_oct) * label_T1_oct
                label_ST1_tone = maskY_ST_1_tone * batch_tone + (1 - maskY_ST_1_tone) * label_T1_tone
                label_ST2_oct = maskY_ST_2_oct * batch_oct + (1 - maskY_ST_2_oct) * label_T2_oct
                label_ST2_tone = maskY_ST_2_tone * batch_tone + (1 - maskY_ST_2_tone) * label_T2_tone

            _, pred_ST3_oct, pred_ST3_tone = StudentB(ST3)
            _, pred_TT_oct, pred_TT_tone = StudentB(TT)

            loss2 =  (0.25 * BCELoss(pred_ST3_oct, label_ST3_oct) + 0.25 * BCELoss(pred_ST3_tone, label_ST3_tone)
                      + 0.25 * BCELoss(pred_TT_oct, label_TT_oct) + 0.25 * BCELoss(pred_TT_tone, label_TT_tone))

            loss2.backward()
            opt2.step()
            train_loss += loss2.item()

        update_ema_variables(StudentB, TeacherB, 0.9999, epoch + 1)

        StudentB.eval()
        eval_arrB = np.zeros(5, dtype=np.double)
        with torch.no_grad():
            for i in range(len(test_list)):
                X_test, y_test = load_data(test_list[i])
                X_test = X_test.to(device)
                pred_list1 = []
                pred_list2 = []
                if gid is not None:
                    split_X = torch.split(X_test, split_size_or_sections=16)
                    for i, split in enumerate(split_X):
                        _, output2, output3 = StudentB(split)
                        pred_list1.append(output2)
                        pred_list2.append(output3)
                    pred_octave = torch.cat(pred_list1, dim=0)
                    pred_tone = torch.cat(pred_list2, dim=0)

                else:
                    pred = StudentB(X_test)
                pred_octave = pred_octave.argmax(dim=1)
                pred_tone = pred_tone.argmax(dim=1)
                est_freq = note2freq(pred_octave, pred_tone).flatten()
                label_octave = freq2octave(y_test)
                label_tone = freq2tone(y_test)
                ref_freq = note2freq(label_octave, label_tone).flatten()
                time_series = np.arange(len(ref_freq)) * 0.01
                eval_arrB += melody_eval(time_series, ref_freq, time_series, est_freq)

            eval_arrB /= len(test_list)

            train_loss /= step + 1


        print("---------Student B------------")
        print("Epoch={:3d}\tTrain_loss={:6.4f}\tLearning_rate={:6.4f}e-4".format(epoch, train_loss, 1e4 *
                                                                                 opt2.state_dict()[
                                                                                     'param_groups'][
                                                                                     0][
                                                                                     'lr']))
        print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arrB[0], eval_arrB[1],
                                                                                       eval_arrB[2], eval_arrB[3],
                                                                                       eval_arrB[4]))
        if eval_arrB[-1] > best_OA_B:
            best_OA_B = eval_arrB[-1]
            best_epoch_B = epoch

        torch.save(StudentB.state_dict(), op + '{:.2f}_{:d}'.format(eval_arrB[4], epoch))
        print('Best Epoch: ', best_epoch_B, ' Best OA: ', best_OA_B)
        print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))




def parser():
    p = argparse.ArgumentParser()

    p.add_argument('-train', '--train_list_path',
                   help='the path of training data list (default: %(default)s)',
                   type=str, default='./train_domain1.txt')
    p.add_argument('-test', '--test_list_path',
                   help='the path of test data list (default: %(default)s)',
                   type=str, default='./train_domain2.txt')
    p.add_argument('-semi', '--semi_list_path',
                   help='the path of semi data list (default: %(default)s)',
                   type=str, default='./train_domain2_semi.txt')
    p.add_argument('-ep', '--epoch_num',
                   help='the number of epoch (default: %(default)s)',
                   type=int, default=300)
    p.add_argument('-bs', '--batch_size',
                   help='The number of batch size (default: %(default)s)',
                   type=int, default=128)
    p.add_argument('-lr', '--learning_rate',
                   help='the number of learn rate (default: %(default)s)',
                   type=float, default=0.0008)
    p.add_argument('-gpu', '--gpu_index',
                   help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s)',
                   type=str, default="0")
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
    pretrained_t = 'pretrained_t.pth'
    pretrained_s = 'pretrained_s.pth'

    seed = random.randint(1, 2**32 - 1)
    #seed = 2781874937
    set_seed(seed)

    if args.gpu_index is not None:
        train(args.train_list_path, args.test_list_path, args.epoch_num, args.batch_size, args.learning_rate,
              args.gpu_index, args.output_dir, args.semi_list_path, pretrained_t, pretrained_s)
    else:
        train(args.train_list_path, args.test_list_path, args.epoch_num, args.batch_size, args.learning_rate,
              args.gpu_index, args.output_dir,  args.semi_list_path, args.pretrained_model)
