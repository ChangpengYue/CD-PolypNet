import os
import sys
import random
import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from PIL import Image
from typing import List, Tuple, Dict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from canny import Net
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
from train.SSFD import SSFDLoss
from utils.logger import create_logger
from efb_net.efbanch import EFBranch

# Helper function for converting float image to uint8 format
def float_to_uint8(image: np.ndarray) -> np.ndarray:
    return (image * 255).astype(np.uint8)

# Canny edge detection wrapper
def canny(raw_img: torch.Tensor, use_cuda: bool = True) -> torch.Tensor:
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(raw_img).cuda() if use_cuda else Variable(raw_img)
    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

    # Save edge detection results
    save_edge_images(grad_mag, thresholded, early_threshold)

    return blurred_img

def save_edge_images(grad_mag, thresholded, early_threshold):
    imageio.imwrite('gradient_magnitude.png', float_to_uint8(grad_mag.data.cpu().numpy()[0, 0]))
    imageio.imwrite('thin_edges.png', float_to_uint8(thresholded.data.cpu().numpy()[0, 0]))
    imageio.imwrite('final.png', float_to_uint8((thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)))
    imageio.imwrite('thresholded.png', float_to_uint8(early_threshold.data.cpu().numpy()[0, 0]))

# Function to redirect stdout and stderr to log file
def redirect_stdout_stderr(log_path: str):
    log_file = open(log_path, "a")
    sys.stdout = log_file
    sys.stderr = log_file

# Save tensor features as text files
def save_feature(tensor: torch.Tensor, name: str, save_path: str):
    tensor = tensor.cpu().detach().numpy()
    os.makedirs(save_path, exist_ok=True)

    for batch_idx in range(tensor.shape[0]):
        for channel_idx in range(tensor.shape[1]):
            feature_map = tensor[batch_idx, channel_idx, :, :]
            filename = os.path.join(save_path, f'{name}_batch{batch_idx}_channel{channel_idx}.txt')
            np.savetxt(filename, feature_map, fmt='%.6f')

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class iAFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # Define local and global attention blocks
        self.local_att = self._create_attention_block(channels, inter_channels)
        self.global_att = self._create_attention_block(channels, inter_channels, global_att=True)
        self.local_att2 = self._create_attention_block(channels, inter_channels)
        self.global_att2 = self._create_attention_block(channels, inter_channels, global_att=True)

        self.sigmoid = nn.Sigmoid()

    def _create_attention_block(self, channels, inter_channels, global_att=False):
        layers = [
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        ]
        if global_att:
            layers.insert(0, nn.AdaptiveAvgPool2d(1))
        return nn.Sequential(*layers)

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type: str):
        super().__init__(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_multimask_outputs=3,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self._load_model_weights(model_type)
        self._initialize_layers()

    def _load_model_weights(self, model_type: str):
        assert model_type in ["vit_b", "vit_l", "vit_h"]
        checkpoint_dict = {
            "vit_b": "pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
            "vit_l": "pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
            "vit_h": "pretrained_checkpoint/sam_vit_h_maskdecoder.pth",
        }
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")

        for n, p in self.named_parameters():
            p.requires_grad = False

    def _initialize_layers(self):
        self.hf_token = nn.Embedding(1, 256)
        self.hf_mlp = MLP(256, 256, 32, 3)
        self.num_mask_tokens += 1
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            LayerNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 32, kernel_size=2, stride=2),
        )
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, 1, 1),
        )
        self.i_aff = iAFF(channels=8)
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.gscnn = EFBranch(2)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, edge, batched_input, image_embeddings, encoder_list_gscnn, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=False, hq_token_only=False, interm_embeddings=None):
        img = batched_input[0]['image'].unsqueeze(0)
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        encoder_list_gscnn_nchw = [tensor.permute(0, 3, 1, 2).contiguous() for tensor in encoder_list_gscnn]
        edge_loss, edge_feature = self.gscnn(encoder_list_gscnn_nchw, img, edge)

        hq_features = self.i_aff(self.embedding_encoder(image_embeddings), self.compress_vit_feat(vit_features))
        hq_features = self.i_aff(edge_feature, hq_features)

        masks, iou_preds = [], []
        for i_batch in range(len(image_embeddings)):
            mask, iou_pred, decoder_list, upscaled_embedding_sam = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                edge_features=edge_feature,
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=hq_features[i_batch].unsqueeze(0),
            )
            masks.append(mask)
            iou_preds.append(iou_pred)

        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)

        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]

        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]
        if hq_token_only:
            return masks_hq, decoder_list, upscaled_embedding_sam, edge_loss
        else:
            return masks_sam, masks_hq, hq_features

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_anns_mask(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious,name):
    if len(masks) == 0:
        return
    if not os.path.exists (filename): 
        os.mkdir (filename)
    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        # plt.imshow(image)
        #print(mask)
        mask_image = show_mask(mask, plt.gca())
        #print(mask_image)
        mask_image = np.asarray(mask_image)
        mask_image = np.squeeze(mask_image)  # Removes singleton dimensions
        mask_image = (mask_image * 255).astype(np.uint8)  # Converts to uint8
        # plt.imshow(mask)
        save_dir = os.path.join(filename,name[0]) + ".png"
        Image.fromarray(mask_image).convert('L').save(save_dir)

        plt.axis('off')
        # plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)
    mask_image = mask.reshape(h, w, 1)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)
    # Adding the necessary arguments
    parser.add_argument("--output", type=str, required=True, help="Path to save masks and checkpoints")
    parser.add_argument("--model-type", type=str, default="vit_l", help="Model type: ['vit_b', 'vit_l', 'vit_h']")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run generation on")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=15, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='Distributed rank')
    parser.add_argument('--local_rank', type=int, help='Local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str, help="Restore checkpoint path")
    parser.add_argument("--gpu", default='cuda:0', type=str, help="CUDA device to use")
    parser.add_argument('--lambda_x', default=0.015, type=float)
    parser.add_argument('--log_path', default='/opt/data/private/ycp/sam-hq-socconv/log/output_all.log', type=str, help='Log file path')
    return parser.parse_args()

# Main function
def main(net, train_datasets, valid_datasets, args):
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Create Data Loaders ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                                my_transforms=[RandomHFlip(), LargeScaleJitter()],
                                                                batch_size=args.batch_size_train, training=True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[Resize(args.input_size)],
                                                           batch_size=args.batch_size_valid, training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- DistributedDataParallel ---
    if torch.cuda.is_available():
        net.cuda()
    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module

    # --- Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model, map_location="cpu"))

        evaluate(args, net, sam, valid_dataloaders, args.visualize)


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)
    
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    kd_loss = KDloss(args.lambda_x)
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        # train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,1000):
            inputs, labels , edge= data['image'], data['label'], data['edge']
            # print(edge)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                edge = edge.cuda()
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                #########这里encoder_list包含了encoder中的四个特征
                batched_output, interm_embeddings, encoder_list, encoder_list_gscnn = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            masks_hq, decoder_list, hq_feature, edge_loss= net(
                edge = edge,
                batched_input = batched_input,
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                encoder_list_gscnn = encoder_list_gscnn,
                # edge_embedding = blurred_img,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )


            loss_edge = edge_loss['edge_loss']
            loss_kd = kd_loss(encoder_list, hq_feature)
            loss_mask, loss_dice,logic = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss = loss_mask + loss_dice + loss_kd + 2*loss_edge
            # loss = loss_mask + loss_dice + loss_edge
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_kd": loss_kd, "loss_edge": loss_edge}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            # 清理不再需要的变量
            del inputs, labels, edge, imgs, batched_input, batched_output, interm_embeddings, encoder_list, encoder_list_gscnn
            del encoder_embedding, image_pe, sparse_embeddings, dense_embeddings, masks_hq, decoder_list, hq_feature, edge_loss
            torch.cuda.empty_cache()  # 清空显存缓存

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, net, sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        net.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.'+key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)



def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori, edge = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'], data_val['edge']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()
            blurred_img = canny(inputs_val)
            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings, encoder_list, encoder_list_gscnn = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            masks_sam, masks_hq , hq_features= net(
                edge = edge,
                batched_input = batched_input,
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                encoder_list_gscnn = encoder_list_gscnn,
                # edge_embedding = blurred_img,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )
            print(hq_features.shape)
            iou = compute_iou(masks_hq,labels_ori)
            # print("IOU:",iou)#####################
            boundary_iou = compute_boundary_iou(masks_hq,labels_ori)

            os.makedirs(args.output, exist_ok=True)
            masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
            for ii in range(len(imgs)):
                base = data_val['im_name']
                # print('base:', base)
                save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                imgs_ii = imgs[ii].astype(dtype=np.uint8)
                show_iou = torch.tensor([iou.item()])
                show_boundary_iou = torch.tensor([boundary_iou.item()])
                show_anns_mask(masks_hq_vis[ii], None, None, None, os.path.join(args.output,'val_mask') , imgs_ii, show_iou, show_boundary_iou,base)
            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats
if __name__ == "__main__":
    
    dataset_polyp_train = {"name": "dataset_eyes_train",
                 "im_dir": "/opt/data/private/ycp/sam-hq-socconv/PolyP_all/train/images",
                 "gt_dir": "/opt/data/private/ycp/sam-hq-socconv/PolyP_all/train/masks",
                 
                 "im_ext": ".png",
                 "gt_ext": ".png"}

    dataset_polyp_val  = {"name": "dataset_eyes_val",
                 "im_dir": "/opt/data/private/ycp/sam-hq-socconv/CVC-ColonDB/images",
                 "gt_dir": "/opt/data/private/ycp/sam-hq-socconv/CVC-ColonDB/masks",
                 "im_ext": ".png",
                 "gt_ext": ".png"}

    train_datasets = [dataset_polyp_train]
    valid_datasets = [dataset_polyp_val] 
    args = get_args_parser()
    net = MaskDecoderHQ(args.model_type) 
    logger = create_logger(args.log_path)
    redirect_stdout_stderr(args.log_path)
    main(net, train_datasets, valid_datasets, args)
