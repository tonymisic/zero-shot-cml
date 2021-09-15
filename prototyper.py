from dataloader import AVE
from utils import GeneralizedZeroShot
from torch.utils.data.dataloader import DataLoader
import os, json, torch, h5py
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# splittings
classes_list_train = json.load(open('AVE_Dataset/classes.json'))
for key in classes_list_train:
    classes_list_train[key] = 0
classes_list_test = json.load(open('AVE_Dataset/classes.json'))
for key in classes_list_train:
    classes_list_test[key] = 0
classes_list_val = json.load(open('AVE_Dataset/classes.json'))
for key in classes_list_train:
    classes_list_val[key] = 0
rootdir = 'AVE_Dataset/'
gzs = GeneralizedZeroShot('AVE_Dataset/', precomputed=True)
gzs.split_precomputed()
ZSL = True
train_data = AVE('AVE_Dataset/', 'train', 'settings.json', precomputed=True, ZSL=ZSL)
train_loader = DataLoader(train_data, 1, shuffle=True, num_workers=1, pin_memory=True)
test_data = AVE('AVE_Dataset/', 'test', 'settings.json', precomputed=True, ZSL=ZSL)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
val_data = AVE('AVE_Dataset/', 'val', 'settings.json', precomputed=True, ZSL=ZSL)
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)

for _, _, temporal_labels, spatial_labels, class_names, back_start, back_end in train_loader:
    for i in class_names:
        if i[0] != 'Background':
            classes_list_train[i[0]] += 1
            break
for _, _, temporal_labels, spatial_labels, class_names, back_start, back_end in test_loader:
    for i in class_names:
        if i[0] != 'Background':
            classes_list_test[i[0]] += 1
            break
for _, _, temporal_labels, spatial_labels, class_names, back_start, back_end in val_loader:
    for i in class_names:
        if i[0] != 'Background':
            classes_list_val[i[0]] += 1
            break
print("training class count")
print(classes_list_train)
print("testing class count")
print(classes_list_test)
print("validation class count")
print(classes_list_val)