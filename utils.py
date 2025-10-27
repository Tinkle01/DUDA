import torch
import numpy as np
import mir_eval
import sys
import time
import librosa
from torch.utils.data import DataLoader, Dataset
import os.path
from torch.autograd import Function
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import lines
import torch.nn.functional as F
import random

LEN_SEG = 64


class Dataset(Dataset):
    def __init__(self, inputs, target, expand=None, mask=None):
        super().__init__()
        self.inputs = inputs
        self.target = target
        self.mask = mask
        self.expand = None
        if expand:
            self.expand = expand
            assert self.expand >= len(self.inputs)

    def __getitem__(self, index):
        if self.expand:
            index = index % len(self.inputs)
        if self.mask is not None:
            return self.inputs[index], self.target[index], self.mask[index]
        return self.inputs[index], self.target[index]

    def __len__(self):
        if self.expand:
            return self.expand
        return len(self.target)


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def to_log(**kwargs):
    log_string = ""
    log_string += f"{kwargs.get('info', '')}:\n" if "info" in kwargs else ""
    log_string += "***********************************\n"

    for key, value in kwargs.items():
        if key != "info":
            log_string += f"{key}:\t{value}\n"

    log_string += "***********************************\n"
    return log_string


def pred2res(pred):
    """
    Convert the output of model to the result
    """
    pred = np.array(pred)
    pred_freq = pred.argmax(axis=1)
    pred_freq[pred_freq > 0] = 31 * 2 ** (pred_freq[pred_freq > 0] / 60)
    return pred_freq


def convert_y(y):
    y[y > 0] = torch.round(torch.log2(y[y > 0] / 31) * 60)
    return y.long()


def y2res(y):
    """
    Convert the label to the result
    """
    y = np.array(y)
    y[y > 0] = 31 * 2 ** (y[y > 0] / 60)
    return y


def melody_eval(ref_time, ref_freq, est_time, est_freq):

    output_eval = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    VR = output_eval["Voicing Recall"] * 100.0
    VFA = output_eval["Voicing False Alarm"] * 100.0
    RPA = output_eval["Raw Pitch Accuracy"] * 100.0
    RCA = output_eval["Raw Chroma Accuracy"] * 100.0
    OA = output_eval["Overall Accuracy"] * 100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr


def load_train_data2(path, cfp_path="cfp",data_path = ""):
    tick = time.time()
    train_list = load_list(path)
    X, y = [], []
    num_seg = 0
    for i in range(len(train_list)):
        print(
            "({:d}/{:d}) Loading data: ".format(i + 1, len(train_list)), train_list[i]
        )
        X_data, y_data = load_data(
            train_list[i], data_path=data_path, cfp_path=cfp_path
        )
        y_data[y_data > 320] = 320
        seg = X_data.size(0)
        num_seg += seg
        X.append(X_data)
        y.append(y_data)
        print(
            "({:d}/{:d})".format(i + 1, len(train_list)),
            train_list[i],
            "loaded: ",
            "{:2d} segments".format(seg),
        )
    print("Data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def load_y(fp, data_path=""):
    with open(data_path + "f0ref/" + fp.replace(".npy", "") + ".txt") as f:
        y = []
        for line in f.readlines():
            y.append(float(line.strip().split()[1]))
        num_seg = len(y) // LEN_SEG
        y = torch.tensor(
            np.array([y[LEN_SEG * i : LEN_SEG * i + LEN_SEG] for i in range(num_seg)]),
            dtype=torch.float32,
        )
        y = convert_y(y)
    return y


def load_multiview_data(
    fp, data_path="", views=[]
):
    fp = fp.split(".")[0]
    y = load_y(fp, data_path=data_path)
    num_seg = y.size(0)

    all_views_npy = []
    for view in views:
        view_npy = np.load(f"{data_path}/{view}/{fp}.npy")
        L = view_npy.shape[2]
        num_seg = min(num_seg, L // LEN_SEG)
        all_views_npy.append(view_npy)

    all_views = []
    for view_npy in all_views_npy:
        view_tensor = torch.tensor(
            np.array(
                [
                    view_npy[:, :, LEN_SEG * i : LEN_SEG * i + LEN_SEG]
                    for i in range(num_seg)
                ]
            ),
            dtype=torch.float32,
        )
        all_views.append(view_tensor)

    return all_views, y[:num_seg]


def load_train_data_with_other_view(path, views=[]):
    data_path = ""
    tick = time.time()
    train_list = load_list(path)
    y = []
    views_list = [[] for _ in range(len(views) + 1)]
    num_seg = 0
    for i in range(len(train_list)):
        print(
            "({:d}/{:d}) Loading data: ".format(i + 1, len(train_list)), train_list[i]
        )

        views_data, y_data = load_multiview_data(
            train_list[i], data_path=data_path, views=views
        )
        for j in range(len(views_data)):
            views_list[j].append(views_data[j])
        y_data[y_data > 320] = 320
        y.append(y_data)

        seg = len(y_data)
        num_seg += seg
        print(
            "({:d}/{:d})".format(i + 1, len(train_list)),
            train_list[i],
            "loaded: ",
            "{:2d} segments".format(seg),
        )
    print("Data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    # return torch.cat(cfp, dim=0), torch.cat(stft, dim=0), torch.cat(y, dim=0)
    return [torch.cat(view, dim=0) for view in views_list], torch.cat(y, dim=0)


@torch.no_grad()
def test_model_with_other_view(net, path, views=[]):
    net.eval()
    test_list = load_list(path)
    eval_arr = np.zeros(5, dtype=np.double)
    for i in range(len(test_list)):
        views_test, y_test = load_multiview_data(
            test_list[i],
            data_path="/media/disk1/dataset/cc_attention2/data/",
            views=views,
        )
        dataset = Dataset(y_test)
        for view in views_test:
            dataset.add_view(view)
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        outputs = []
        for inputs, _ in loader:
            pred = net(inputs)
            outputs.append(pred[-1])
        est_freq = torch.cat(outputs, 0)
        est_freq = pred2res(est_freq.cpu()).flatten()
        ref_freq = y2res(y_test).flatten()
        time_series = np.arange(len(ref_freq)) * 0.01
        eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)
    eval_arr /= len(test_list)
    return eval_arr


@torch.no_grad()
def test_model(net, tonet,  path, log_file=None, idx=None):
    net.eval()
    test_list = load_list(path)
    for test in test_list:
        test = test.split(".")[0]
        if not os.path.exists(f"wavs/{test}.wav"):
            os.system(f"cp /{test}.wav ./wavs/")
    eval_arr = np.zeros(5, dtype=np.double)
    count = 0
    print("assigning idx: ", idx)
    for i in range(len(test_list)):
        count += 1
        if idx is not None:
            i = idx
        print(test_list[i])
        X_test, y_test = load_data(
            test_list[i],
            data_path="",
        )
        dataset = Dataset(X_test, y_test)
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        outputs = []
        for x, _ in loader:
            with torch.no_grad():
                _, _, _, feature_oct, feature_tone = tonet(x)
            pred, feat_f0 = net(x,feature_oct, feature_tone, return_feat=True)
            outputs.append(pred)
        est_freq = torch.cat(outputs, 0)
        est_freq = pred2res(est_freq.cpu()).flatten()
        ref_freq = y2res(y_test).flatten()
        if log_file:
            with open(log_file, "w") as f:
                for est, ref in zip(est_freq, ref_freq):
                    f.write(f"{est}\t{ref}\n")
        time_series = np.arange(len(ref_freq)) * 0.01
        eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)
        if idx is not None:
            break
    eval_arr /= count
    return eval_arr


def flatten_2d(x):
    """
    Flatten the torch.Size([bs, dim, time]) tensor to torch.Size([bs*time, dim] tensor
    """

    return x.permute(0, 2, 1).reshape(-1, x.size(1))

if __name__ == "__main__":
    import torch.nn.functional as F
    import ot
    def pairwise_cosine_sim(a, b):
        assert len(a.shape) == 2
        assert a.shape[1] == b.shape[1]
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        mat = a @ b.t()
        return mat
    def ot_mapping(M):
        """
        M: (ns, nt)
        """
        reg1 = 1
        reg2 = 1
        ns, nt = M.shape
        a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
        gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
        gamma = torch.from_numpy(gamma).cuda()
        return gamma
    proto=torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float32).cuda()
    feat_tu_w=torch.tensor([[1,2,3],[4,5,6],[1,2,6],[10,11,12],[1,5,12]],dtype=torch.float32).cuda()
    feat_tu_s = torch.tensor([[2,2,3],[4,6,6],[1,1,6],[6,11,12],[4,5,12]],dtype=torch.float32).cuda()
    def contras_cls(p1, p2):
        N, C = p1.shape
        cov = p1.t() @ p2

        print(cov)
        cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
        print(cov_norm1)
        cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
        print(cov_norm2)
        loss = (
            0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C
            + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
        )
        return loss
    # print(contras_cls(feat_tu_w,feat_tu_s))
    M = 1 - pairwise_cosine_sim(
        proto, feat_tu_w
    )
    print(M)
    gamma_st_weak = ot_mapping(M.data.cpu().numpy().astype(np.float64))
    print(gamma_st_weak)
    labels = gamma_st_weak.t().argmax(1)
    print(labels)
    labels = labels.unsqueeze(1).expand(5, 3)
    classes = torch.arange(3).long().expand(5, 3).cuda()
    proto = F.normalize(proto)
    feat_tu_s = F.normalize(feat_tu_s)
    distmat = -feat_tu_s @ proto.t() + 1
    mask = labels.eq(classes)
    dist  =distmat * mask.float()
    print(dist)
    loss=dist.sum()/5
    print(loss)



def freq2octave(freq_bin):
    octave = torch.where(
        freq_bin == 0, 0,
        torch.ceil(freq_bin / 60)
    )

    return octave.long()


def freq2tone(freq_bin):
    tone = torch.where(
        freq_bin == 0, 0,
        torch.ceil(freq_bin / 5) % 12
    )

    tone = torch.where(
        (tone == 0) & (freq_bin != 0), 12,
        tone
    )
    return tone.long()


def note2freq(octave, tone):
    freq_bin = torch.where(
        (octave == 0) | (tone == 0), 0,
        (octave - 1) * 60 + tone * 5
    )

    return np.array(freq_bin.cpu())


def load_list(path, mode):
    if mode == 'test':
        f = open(path, 'r')
    elif mode == 'train':
        f = open(path, 'r')
    else:
        raise Exception("mode must be 'test' or 'train'")
    data_list = []
    for line in f.readlines():
        data_list.append(line.strip())
    if mode == 'test':
        print("{:d} test files: ".format(len(data_list)))
    else:
        print("{:d} train files: ".format(len(data_list)))
    return data_list


def load_train_data(path, mode='train'):
    tick = time.time()
    train_list = load_list(path, mode='train')
    X, y = [], []
    num_seg = 0
    for i in range(len(train_list)):
        print('({:d}/{:d}) Loading data: '.format(i + 1, len(train_list)), train_list[i])
        if mode == 'train':
            X_data, y_data = load_data(train_list[i])
        else:
            X_data, y_data = load_data(train_list[i], 'test')
        y_data[y_data > 320] = 320
        seg = X_data.size(0)
        num_seg += seg
        X.append(X_data)
        y.append(y_data)
        print('({:d}/{:d})'.format(i + 1, len(train_list)), train_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Training data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def load_semi_data(path):
    tick = time.time()
    train_list = load_list(path, mode='train')
    X = []
    num_seg = 0
    for i in range(len(train_list)):
        print('({:d}/{:d}) Loading data: '.format(i + 1, len(train_list)), train_list[i])
        X_data = load_onlyx_data(train_list[i])
        seg = X_data.size(0)
        num_seg += seg
        X.append(X_data)
        print('({:d}/{:d})'.format(i + 1, len(train_list)), train_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Training data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    return torch.cat(X, dim=0)



def load_data(fp, mode='train'):
    '''
    X: (N, C, F, T)
    y: (N, T)
    '''
    X = np.load('' + fp)
    L = X.shape[2]
    num_seg = L // LEN_SEG
    X = torch.tensor(np.array([X[:, :, LEN_SEG * i:LEN_SEG * i + LEN_SEG] for i in range(num_seg)]),
                     dtype=torch.float32)

    f = open('' + fp.replace('.npy', '') + '.txt')
    y = []
    for line in f.readlines():
        y.append(float(line.strip().split()[1]))
    num_seg = min(len(y) // LEN_SEG, num_seg)
    y = torch.tensor(np.array([y[LEN_SEG * i:LEN_SEG * i + LEN_SEG] for i in range(num_seg)]), dtype=torch.float32)
    if mode == 'train':
        y = convert_y(y)

    return X[:num_seg], y[:num_seg]


def load_onlyx_data(fp, mode=320):
    '''
    X: (N, C, F, T)
    y: (N, T)
    '''
    if mode == 320:
        load_path = '' + fp
        if os.path.exists(load_path):
            X = np.load(load_path)
        else:
            X = np.load('' + fp)
    elif mode == 64:
        X = np.load('' + fp)
    elif mode == 128:
        X = np.load('' + fp)
    L = X.shape[2]
    num_seg = L // LEN_SEG
    X = torch.tensor([X[:, :, LEN_SEG * i:LEN_SEG * i + LEN_SEG] for i in range(num_seg)], dtype=torch.float32)

    return X[:num_seg]



def load_domain_data(path1, path2):
    source_list = load_list(path1, 'train')
    target_list = load_list(path2, 'train')
    X, y = [], []
    num_seg = 0
    tick = time.time()

    for i in range(len(source_list)):
        print('({:d}/{:d}) Loading source data: '.format(i + 1, len(source_list)), source_list[i])

        X_data, y_data = mark_label(source_list[i], 0)
        X.append(X_data)
        y.append(y_data)
        seg = X_data.size(0)
        num_seg += seg

        print('({:d}/{:d})'.format(i + 1, len(source_list)), source_list[i], 'loaded: ', '{:2d} segments'.format(seg))

    for i in range(len(target_list)):
        print('({:d}/{:d}) Loading target data: '.format(i + 1, len(target_list)), target_list[i])

        X_data, y_data = mark_label(target_list[i], 1)
        X.append(X_data)
        y.append(y_data)

        seg = X_data.size(0)
        num_seg += seg

        print('({:d}/{:d})'.format(i + 1, len(target_list)), target_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Domain data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))

    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def load_domain_data2(X_train, path2):
    target_list = load_list(path2, 'train')
    X, y = [], []
    num_seg = 0
    tick = time.time()

    y_data = mark_label2(X_train, 0)
    y.append(y_data)
    seg = X_train.size(0)
    num_seg += seg

    for i in range(len(target_list)):
        print('({:d}/{:d}) Loading target data: '.format(i + 1, len(target_list)), target_list[i])

        X_data, y_data = mark_label(target_list[i], 1)
        X.append(X_data)
        y.append(y_data)

        seg = X_data.size(0)
        num_seg += seg

        print('({:d}/{:d})'.format(i + 1, len(target_list)), target_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Domain data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))

    out = torch.cat((X_train, torch.cat(X, dim=0)), dim=0)

    return out, torch.cat(y, dim=0)



def mark_label(path, domain):
    X = np.load('' + path)
    L = X.shape[2]
    num_seg = L // LEN_SEG
    X = torch.tensor(np.array([X[:, :, LEN_SEG * i:LEN_SEG * i + LEN_SEG] for i in range(num_seg)]),
                     dtype=torch.float32)

    if domain == 0:
        y = torch.zeros((num_seg,), dtype=torch.int64)
    else:
        y = torch.ones((num_seg,), dtype=torch.int64)

    return X, y


def mark_label2(X_train, domain):
    if domain == 0:
        y = torch.zeros((X_train.size(0),), dtype=torch.int64)
    else:
        y = torch.ones((X_train.size(0),), dtype=torch.int64)

    return y

def f02img(y):
    N = y.size(0)
    img = torch.zeros([N, 321, LEN_SEG], dtype=torch.float32)
    for i in range(N):
        img[i, y[i], torch.arange(LEN_SEG)] = 1
    return img


def octave2img(y):
    N = y.size(0)
    img = torch.zeros([N, 7, LEN_SEG], dtype=torch.float32)
    for i in range(N):
        img[i, y[i], torch.arange(LEN_SEG)] = 1
    return img


def tone2img(y):
    N = y.size(0)
    img = torch.zeros([N, 13, LEN_SEG], dtype=torch.float32)
    for i in range(N):
        img[i, y[i], torch.arange(LEN_SEG)] = 1
    return img


def pos_weight(data):
    N = data.size(0)
    non_melody = torch.sum(data[:, 0, :]).item() + 1
    melody = (N * LEN_SEG) - non_melody + 2
    z = torch.zeros((321, LEN_SEG), dtype=torch.float32)

    z[1:, :] += non_melody / melody
    z[0, :] += melody / non_melody
    return z


def ce_weight(data):
    N = data.size(0)
    non_melody = torch.sum(data == 0) + 1
    melody = (N * LEN_SEG) - non_melody + 2
    z = torch.zeros(321, dtype=torch.float32)
    z[1:] += non_melody / melody
    z[0] += melody / non_melody
    return z


def build_harmonic3(y):

    harmonic = torch.zeros_like(y)
    sub_harmonic = torch.zeros_like(y)
    non_matrix = torch.ones_like(y)
    index = torch.argmax(y, dim=1)

    index_harmonic = [torch.where((index + 60 < 321) & (index != 0), index + 60, -1),
                      ]

    index_sub_harmonic = [torch.where((index - 60 > 0) & (index != 0), index - 60, -1),
                          ]
    for i in range(y.size(0)):
        for k in range(y.size(2)):

            a = index[i, k]
            if y[i, a, k] == 1:
                non_matrix[i, a, k] = 0

            for t in range(len(index_sub_harmonic)):
                b = index_harmonic[t][i, k]
                c = index_sub_harmonic[t][i, k]

                if b != -1:
                    harmonic[i, b, k] = 1
                    non_matrix[i, b, k] = 0

                if c != -1:
                    sub_harmonic[i, c, k] = 1
                    non_matrix[i, c, k] = 0

    output = torch.cat(
        (y.unsqueeze(dim=0), harmonic.unsqueeze(dim=0), sub_harmonic.unsqueeze(dim=0), non_matrix.unsqueeze(dim=0)),
        dim=0)

    return output.permute(1, 0, 2, 3)

def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Update EMA variables of the model.

    Args:
        model (torch.nn.Module): The model to be updated.
        ema_model (torch.nn.Module): The EMA model.
        alpha (float): The decay rate.
        global_step (int): The global step.
    """
    #alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)










