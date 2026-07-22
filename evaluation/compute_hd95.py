
import os, torch, numpy as np
from torch.utils.data import DataLoader
from models.he_friendly import HEFriendlyUNet
from training.dataset import ACDCDataset

device = torch.device('mps')
model = HEFriendlyUNet(act_type='poly', norm_type='batch').to(device)
model.load_state_dict(torch.load('results/poly_batch_nofilter/act=poly_norm=batch_bs8_lr0.0001/best_model.pth', 
                                  map_location='cpu', weights_only=False))
model.eval()

data_dir = os.path.expanduser('~/Desktop/tesi_acdc/training')
patients = sorted(os.listdir(data_dir))
all_cases = []
for p in patients:
    pdir = os.path.join(data_dir, p)
    if not os.path.isdir(pdir): continue
    for f in os.listdir(pdir):
        if f.endswith('.nii.gz') and '_gt' not in f and '_4d' not in f:
            frame = f.replace(p + '_frame', '').replace('.nii.gz', '')
            all_cases.append(f'{p}_frame{frame}')
n = len(all_cases)
val_cases = all_cases[int(n*0.8):]

ds = ACDCDataset(data_dir, val_cases, patch_size=(256,224), augment=False)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

dice_rv, dice_myo, dice_lv = [], [], []
with torch.no_grad():
    for imgs, segs in loader:
        imgs, segs = imgs.to(device), segs.to(device)
        preds = model(imgs).argmax(dim=1)
        for label, lst in [(1,dice_rv),(2,dice_myo),(3,dice_lv)]:
            p2 = (preds==label).float()
            t = (segs==label).float()
            score = (2*(p2*t).sum()+1e-5)/(p2.sum()+t.sum()+1e-5)
            lst.append(score.item())

print(f'RV:  {sum(dice_rv)/len(dice_rv):.3f}')
print(f'MYO: {sum(dice_myo)/len(dice_myo):.3f}')
print(f'LV:  {sum(dice_lv)/len(dice_lv):.3f}')
print(f'Mean: {(sum(dice_rv)+sum(dice_myo)+sum(dice_lv))/(3*len(dice_rv)):.3f}')
