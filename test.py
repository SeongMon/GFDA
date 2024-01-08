import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.metrics import cal_dice, cal_iou, cal_wfm, cal_sm, cal_em, cal_mae
import numpy as np

import data.CustomDataset
import data.LoadData
from tqdm import tqdm
from model.PraNet import PraNet
from data.LoadData import test_dataset_Kvasir

def custom_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = 'trained_gfda.pth'
loaded_model = PraNet()  # Replace 'PraNet' with the actual class of your model
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)  # Move the loaded model to the device (GPU or CPU)
loaded_model.eval()

test_loader_Kvasir = DataLoader(test_dataset_Kvasir, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
with torch.no_grad():   
    total_loss = 0
    iou_score = 0
    Dice_score = 0
    accuracy = 0
    recall = 0 
    precision = 0
    F1score = 0
    F2score = 0

    #output_directory1 = 'result_image/pred_0.00001'
    #output_directory2 = 'result_image/mask'
    idx = 0

    m_dice, m_iou, wfm, sm, em, mae = cal_dice(), cal_iou(), cal_wfm(), cal_sm(), cal_em(), cal_mae()
    tbar = tqdm(test_loader_Kvasir, total = len(test_loader_Kvasir), leave = True)
    for images, masks in tbar:    
        images = images.to(device)  # device는 GPU 장치를 가리키는 변수입니다.
        masks = masks.to(device)
        images = images.float() # convert to torch.float32
        masks = masks.float()
        res5, res4, rest3, output  = loaded_model(images)

        criterion = nn.BCEWithLogitsLoss()
        
        masks = torch.squeeze(masks, dim=0)
        pred2 = torch.squeeze(output, dim=0)
        masks = torch.squeeze(masks, dim=0)
        pred2 = torch.squeeze(pred2, dim=0)

        pred2 = torch.sigmoid(pred2)
        pred2 = (pred2 >= 0.5)
        pred2 = pred2.cpu()
        pred2 = pred2.detach().numpy()
        
        masks = masks.cpu()
        masks = masks.detach().numpy()

        gt = np.asarray(masks, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0


        #pred2_binary = (pred2 * 255).astype(np.uint8)
        #output_filename1 = os.path.join(output_directory1, f'prediction_{idx}.png')
        #cv2.imwrite(output_filename1, pred2_binary)

        # gt_binary = (gt * 255).astype(np.uint8)
        # output_filename2 = os.path.join(output_directory2, f'gt_{idx}.png')
        # cv2.imwrite(output_filename2, gt_binary)

        res = pred2
        res = np.array(res)
        if res.max() == res.min():
            res = res/255
        else:
            res = (res ^ res.min()) / (res.max() ^ res.min())
        mae.update(res, gt)
        sm.update(res,gt)
        em.update(res, gt)
        wfm.update(res,gt)
        m_dice.update(res, gt)
        m_iou.update(res, gt)
        idx+=1
        
    MAE = mae.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    print('dataset: test_Kvasir M_dice: {:.4f} M_iou: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} MAE: {:.4f}'
          .format(m_dice, m_iou,wfm,sm,em,MAE))