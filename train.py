import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import islice
from tqdm import tqdm
from model.PraNet_Res2Net import PraNet
from model.gfda import create_fda_gau
from data.LoadData import train_dataset_Kvasir, train_dataset_CVC


lr = 1e-4
batch_size = 4
epochs = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#for control randomness
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)

def custom_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks
train_loader_CVC = DataLoader(train_dataset_CVC, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
train_loader_Kvasir = DataLoader(train_dataset_Kvasir, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

net = PraNet().to(device)

optimizer = torch.optim.Adam(net.parameters() , lr=lr, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
train_losses = []
val_losses = []
start_epoch = 0


def entropy_loss(logits):
    probs = torch.nn.functional.softmax(logits, dim=1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy

def structure_loss(pred, mask): #pranet loss
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

train_losses = []
train_fda_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
dice_arr = []
iou_arr = []
fn_loss = nn.BCEWithLogitsLoss().to(device)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    net.train()
    loss_arr = []
    return_mask = []
    train_loss = 0.0
    trg_dice = 0.0
    fda_loss = 0.0
    fda_loss_arr = []
    tbar_CVC = tqdm(train_loader_CVC, total = len(train_loader_CVC), leave = True, position=0)
    tbar_Kvasir = tqdm(train_loader_Kvasir, total = len(train_loader_Kvasir), leave = True, position=1)

    for i, ((images_Kvasir, masks_Kvasir),(images_CVC, masks_CVC)) in enumerate( zip(tbar_Kvasir, islice( tbar_CVC, len(tbar_Kvasir) ) )):
        images_CVC = images_CVC.to(device)
        masks_CVC = masks_CVC.to(device)
        images_CVC = images_CVC.float()
        masks_CVC = masks_CVC.float()

        images_Kvasir = images_Kvasir.to(device)
        masks_Kvasir = masks_Kvasir.to(device)
        images_Kvasir = images_Kvasir.float()
        masks_Kvasir = masks_Kvasir.float()

        optimizer.zero_grad()
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = net(images_CVC)
        loss5 = structure_loss(lateral_map_5,masks_CVC)
        loss4 = structure_loss(lateral_map_4,masks_CVC)
        loss3 = structure_loss(lateral_map_3,masks_CVC)
        loss2 = structure_loss(lateral_map_2,masks_CVC)
        loss_src = loss2 + loss3 + loss4 + loss5 

        src_in_trg = create_fda_gau(images_Kvasir, images_CVC, 0.000045,0.000045)
        src_in_trg = src_in_trg.to(device)
        src_in_trg = src_in_trg.float()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = net(src_in_trg)
        loss5 = structure_loss(lateral_map_5,masks_CVC)
        loss4 = structure_loss(lateral_map_4,masks_CVC)
        loss3 = structure_loss(lateral_map_3,masks_CVC)
        loss2 = structure_loss(lateral_map_2,masks_CVC)
        loss_trg = loss2 + loss3 + loss4 + loss5 

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = net(images_Kvasir)
        loss5 = entropy_loss(lateral_map_5)
        loss4 = entropy_loss(lateral_map_4)
        loss3 = entropy_loss(lateral_map_3)
        loss2 = entropy_loss(lateral_map_2)
        loss_trg_ent = loss2 + loss3 + loss4 + loss5 

        loss_sum = loss_src + loss_trg + loss_trg_ent
        loss_sum.backward()
        optimizer.step()

        train_loss += loss_sum.item()
        loss_arr += [loss_sum.item()]

        tbar_CVC.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_CVC), len(train_loader_CVC.dataset),
            100. * i / len(train_loader_CVC), (train_loss/(i+1) ) ))

    scheduler.step()  
    train_losses.append(np.mean(loss_arr))

torch.save(net.state_dict(), 'trained_gfda.pth')