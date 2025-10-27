import os
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=4e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=129065)
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--mother_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default="")
parser.add_argument("--threshold", type=float, default=0.95)
parser.add_argument("--source", type=str, default="1010")
parser.add_argument("--target", type=str, default="52")
parser.add_argument("--ot_weight", type=float, default=0)
parser.add_argument("--con_cls_weight", type=float, default=0)
parser.add_argument("--fix_weight", type=float, default=0)
parser.add_argument("--testlist", type=str, default="test_classical_npy.txt")
args = parser.parse_args()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
import numpy as np
from tqdm import tqdm
import time
from utils import *
from model import *
from losses import *
from TO_msnet import MSNet as TO_MSNet

save_dir = None

def main():
    lr = args.lr
    batch_size = args.batch_size
    seed = args.seed
    device = "cuda"
    # seed = torch.randint(0, torch.iinfo(torch.int32).max, (1,)).item()
    seed = random.randint(1,2**32-1)
    print('seed:', seed)
    set_seed(seed)
    info = to_log(
        info="Hyper Param",
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        args=args,
    )
    print(info)
    global save_dir
    if args.mother_dir is None:
        mother_dir = ''
        save_dir = ''
        mother_dir = mother_dir.strip()
        save_dir = save_dir.strip()
    else:
        mother_dir = args.mother_dir
        save_dir = args.save_dir
    save_dir = mother_dir + "/" + save_dir
    save_dir = './model_fusion'

    if "test" not in save_dir:
        save_dir += f"{args.threshold}_{lr:.0e}_{batch_size}_{seed}"

    model_dir = save_dir + "/model"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.system(f"cp {sys.argv[0]} {save_dir}")
    os.system(f"cp startup.sh {save_dir}")
    os.system(f"cp model.py {save_dir}")
    os.system(f"cp utils.py {save_dir}")
    os.system(f"cp losses.py {save_dir}")
    os.system(f"cp startup.sh {save_dir}")
    with open(save_dir + "/info.txt", "w+") as f:
        f.write(info)
    print(f"All files are saved in {save_dir}")
    start = time.time()
    X_train_source, y_train_source = load_train_data2(
        path="jazz_npy2.txt", cfp_path="cfp"
    )
    X_train_target_weak, _ = load_train_data2(
        path="opera_npy12.txt", cfp_path="cfp_weak"
    )
    X_train_target_strong, _ = load_train_data2(
        path="opera_npy12.txt", cfp_path="cfp_strong"
    )
    y_train_source = f02img(y_train_source)

    with open(save_dir + "/info.txt", "a") as f:
        print("source ", X_train_source.shape, file=f)
        print("target weak ", X_train_target_weak.shape, file=f)
        print("target strong ", X_train_target_strong.shape, file=f)
    print(f"Loading data takes {time.time() - start:.2f}")
    source_dataset = Dataset(
        X_train_source, y_train_source, expand=max(len(X_train_source),len(X_train_target_weak))
    )
    target_dataset = Dataset(X_train_target_weak, X_train_target_strong, expand=max(len(X_train_source),len(X_train_target_weak)))
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    device = "cuda:0"
    net = MSNet().to(device)
    tonet = TO_MSNet().to(device)

    tonet = torch.nn.DataParallel(tonet, device_ids=[0])


    tonet.load_state_dict(torch.load('model.pth'), strict=True)
    proto_s = Prototype(321, 321)
    with open(save_dir + "/info.txt", "a") as f:
        print(net, file=f)
        print("source dataset: ",len(source_dataset), file=f)
        print("target dataset: ",len(target_dataset), file=f)
    ENDURANCE = 30
    endurance = ENDURANCE

    best = {
        "ADC2004": [[0, 0, 0, 0, 0], 0],
        "MIREX05": [[0, 0, 0, 0, 0], 0],
        "MedleyDB": [[0, 0, 0, 0, 0], 0],
        "OPERA": [[0, 0, 0, 0, 0], 0],
    }
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start = time.time()
    for epoch in range(200):
        total_losses = []
        ce_losses = []
        ot_losses = []
        con_cls_losses = []
        fix_losses = []
        temp_losses = []
        net.train()
        with tqdm(total=len(source_loader), desc=f"Epoch {epoch}", unit="step") as pbar:
            for (X_s, y_s), (X_t_w, X_t_s) in zip(source_loader, target_loader):
                with torch.no_grad():
                    _,_,_,feature_oct, feature_tone = tonet(X_s)
                y_s = y_s.to(device)
                pred_s, feat_s = net(X_s, feature_oct, feature_tone, return_feat=True)
                pred_t_w, feat_t_w = net(X_t_w,feature_oct, feature_tone, return_feat=True)
                pred_t_s, feat_t_s = net(X_t_s, feature_oct, feature_tone,return_feat=True)

                pred_s = flatten_2d(pred_s)
                feat_s = flatten_2d(feat_s)

                y_s = flatten_2d(y_s)
                y_s = y_s.argmax(dim=1)
                L_ce = F.cross_entropy(feat_s, y_s)
                pred_t_w = flatten_2d(pred_t_w)
                pred_t_s = flatten_2d(pred_t_s)
                feat_t_w = flatten_2d(feat_t_w)
                feat_t_s = flatten_2d(feat_t_s)
                
                L_ot, L_con_cls, L_fix, consis_mask = loss_unl(
                    pred_t_w, feat_t_w, pred_t_s, feat_t_s, proto_s, args
                )
                # L_ot *= 2
                # L_con_cls *= args.con_cls_weight
                # L_con_cls *= 0.1
                # L_fix *= 0.1
                loss = L_ce + L_ot + L_con_cls + L_fix
                opt.zero_grad()
                loss.backward()
                opt.step()

                proto_s.update(feat_s, y_s)
                total_losses.append(loss.item())
                ce_losses.append(L_ce.item())
                ot_losses.append(L_ot.item())
                con_cls_losses.append(L_con_cls.item())
                fix_losses.append(L_fix.item())
                pbar.set_postfix(
                    {
                        "all": f"{(loss.item()):.4f}",
                        "ce": f"{L_ce:.4f}",
                        "ot": f"{L_ot:.4f}",
                        "con_cls": f"{L_con_cls:.4f}",
                        "fix": f"{L_fix:.4f}",
                    }
                )
                pbar.update(1)
        now = time.time()
        log = to_log(
            info=f"Epoch {epoch}",
            total_loss=np.mean(total_losses),
            ce_loss=np.mean(ce_losses),
            con_cls_loss=np.mean(con_cls_losses),
            fix_loss=np.mean(fix_losses),
            time=now - start,
        )
        print(log)

        now = time.time()
        VR, VFA, RPA, RCA, OA = test_model(net,tonet, args.testlist)
        if OA > best["MIREX05"][0][0]:
            best["MIREX05"] = [[OA, VFA, RPA, RCA, VR], epoch]
            net.save(f"{model_dir}/{OA:.2f}_{epoch}_op.pth")
            endurance = ENDURANCE
        metric = to_log(
            info=f"test in MIREX05, time: {time.time() - now:.2f}",
            OA=f"{OA:.2f}",
            VFA=f"{VFA:.2f}",
            RPA=f"{RPA:.2f}",
            RCA=f"{RCA:.2f}",
            VR=f"{VR:.2f}\n",
            BEST=f"{best['MIREX05'][0][0]:.2f} in epoch {best['MIREX05'][1]}",
        )
        log += metric
        print(metric)

        with open(save_dir + "/log.txt", "a+") as f:
            f.write(log + "\n")

        endurance -= 1
        if endurance < 0:
            break
    with open(save_dir + "/info.txt", "a+") as f:
        f.write("\nBest\n")
        f.write("\tOA\tVFA\tRPA\tRCA\tVR\tEpoch\n")
        for key, value in best.items():
            f.write(key + "\t")
            for v in value[0]:
                f.write(f"{v:.2f}\t")
            f.write(f"{value[1]}\n")
    sys.exit(0)


import traceback

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        sys.exit(e.code)
    except BaseException as e:
        with open(save_dir + "/info.txt", "a+") as f:
            error_info = traceback.format_exc()
            print(error_info)
            print(error_info, file=f)
        sys.exit(1)
