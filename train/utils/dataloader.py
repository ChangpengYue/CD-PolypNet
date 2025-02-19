# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
import utils.edge_utils as edge_utils
#### --------------------- dataloader online ---------------------####

palette = [
    0, 0, 0,      # 类别1的颜色 (黑色)，对应0
    255, 255, 255 # 类别2的颜色 (白色)，对应255
]

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        # sampler = DistributedSampler(gos_dataset)
        # batch_sampler_train = torch.utils.data.BatchSampler(
        #     sampler, batch_size, drop_last=True)
        # dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)
        dataloader = DataLoader(gos_dataset,  batch_size, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            # sampler = DistributedSampler(gos_dataset, shuffle=False)
            # dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)
            dataloader = DataLoader(gos_dataset,  batch_size, num_workers=num_workers_, shuffle=True)
            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, edge, shape =  sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']

        # random horizontal flip

        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])
            edge = torch.flip(edge,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'edge':edge, 'shape':shape}

class Resize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        # imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        imidx, image, label, edge, shape =  sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)
        edge = torch.squeeze(F.interpolate(torch.unsqueeze(edge,0),self.size,mode='bilinear'),dim=0)

        # return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}
        return {'imidx':imidx,'image':image, 'label':label, 'edge':edge, 'shape':torch.tensor(self.size)}


class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        # imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        imidx, image, label, edge, shape =  sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']


        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]
        edge = edge[:,top:top+new_h,left:left+new_w]

        # return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}
        return {'imidx':imidx,'image':image, 'label':label, 'edge':edge, 'shape':torch.tensor(self.size)}



class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        # imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        imidx, image, label, edge, shape =  sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']

        image = normalize(image,self.mean,self.std)

        # return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}
        return {'imidx':imidx,'image':image, 'label':label, 'edge':edge, 'shape':shape}



# class LargeScaleJitter(object):
#     """
#         Implementation of large scale jitter from copy_paste
#         https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
#     """

#     def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
#         self.desired_size = torch.tensor(output_size)
#         self.aug_scale_min = aug_scale_min
#         self.aug_scale_max = aug_scale_max

#     def pad_target(self, padding, target):
#         target = target.copy()
#         if "masks" in target:
#             target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
#         return target

#     def __call__(self, sample):
#         imidx, image, label, image_size = sample['imidx'], sample['image'], sample['label'], sample['shape']

#         # Resize while keeping ratio
#         out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

#         random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
#         scaled_size = (random_scale * self.desired_size).round()

#         scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
#         scaled_size = (image_size * scale).round().long()

#         scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), scaled_size.tolist(), mode='bilinear'), dim=0)
#         scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), scaled_size.tolist(), mode='bilinear'), dim=0)

#         # Remove random crop logic, keeping the image and label as they are after scaling

#         # Pad the image and label to the desired size
#         padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
#         padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
#         image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
#         label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)

#         return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(image.shape[-2:])}

class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        # imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        imidx, image, label, edge, image_size =  sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']


        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min          
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_edge = torch.squeeze(F.interpolate(torch.unsqueeze(edge,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        
        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_edge = scaled_edge[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)
        edge = F.pad(scaled_edge, [0,padding_w, 0,padding_h],value=0)

        return {'imidx':imidx,'image':image, 'label':label, 'edge':edge, 'shape':torch.tensor(image.shape[-2:])}

class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im_name = self.dataset["im_name"][idx]    
        im = np.asarray(Image.open(im_path).convert('RGB')) 
        gt = np.asarray(Image.open(gt_path).convert('L')) 
        gt = np.copy(gt)

        # print(gt.shape)
        # gt[gt == 255 ] = 1


        image = Image.fromarray(im.astype(np.uint8))

        _edgemap = edge_utils.mask_to_onehot(gt, 2)

        # _edgemap(1, 288, 384) 二值图
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
        ##删掉一维
        # _edgemap = _edgemap[0, :, :]
        edgemap = torch.from_numpy(_edgemap).float()
        #########边缘分割的很完美

##############   debug   ###################
        # _edgemap = _edgemap[0, :, :]
        # _edgemap[_edgemap==1] = 255
        # edge = Image.fromarray(_edgemap)

        # outdir = '../dump_imgs_{}'.format('val')
        # os.makedirs(outdir, exist_ok=True)
        # out_img_fn = os.path.join(outdir, im_name + '.png')
        # out_msk_fn = os.path.join(outdir, im_name + '_mask.png')
        # out_edge_fn = os.path.join(outdir, im_name + '_edge.png')
        # mask_img = colorize_mask(np.array(gt))
        # image.save(out_img_fn)
        # mask_img.save(out_msk_fn)
        # edge.save(out_edge_fn)
        # print(out_msk_fn)

        # dataloader for bi image
        # if gt.max() == 1:
        #     gt = gt * 255
        #print(f"Before gt : {gt}, unique value: {np.unique(gt)}")
        # 1,2 眼白 # 3,4 瞳孔

        # gt[gt ==2 ] = 255
        # gt[gt ==3 ] = 255
        # gt[gt ==4 ] = 255
        #print(f"gt : {gt}, unique value: {np.unique(gt)}")

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)
        #print(f"last gt : {gt}, unique value: {np.unique(gt)}")


        # _edgemap = gt.numpy()

        
# image.shape: torch.Size([3, 288, 384])
# label.shape: torch.Size([1, 288, 384])
# edge.shape: torch.Size([288, 384])

        # print("image.shape:", im.shape)
        # print("label.shape:", gt.shape)
        # print("edge.shape:", edgemap.shape)
        

        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "label": gt,
        "edge": edgemap,
        "shape": torch.tensor(im.shape[-2:]),
        "imname": im_name,
        }
        
        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]
            sample['im_name'] = self.dataset["im_name"][idx]
 
        return sample