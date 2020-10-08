"""ImageEmbedder model code.

Portions adapted from https://github.com/macaodha/inat_comp_2018/blob/master/train_inat.py
and https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
"""
from datetime import datetime
from PIL import Image
import json
import logging
import math
import numpy as np
import os
import random
import sys
import time
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

import celeba_reader
import evaluation
import inat_reader
from model_utils import *

class ImageDataset(Dataset):
    """An image dataset

    If is_pairs=True, then raw_data is List of (id1, id2, label)
    if is_pairs=False, then raw_data is just List of image IDs
    """
    def __init__(self, raw_data, is_pairs):
        self.raw_data = raw_data
        self.is_pairs = is_pairs
        #self.im_size = [299, 299]  # Inception v3
        self.im_size = [224, 224]  # ResNet
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)


    def _process_image(self, image_id):
        raise NotImplementedError

    def __getitem__(self, index):
        if self.is_pairs:
            id1, id2, label = self.raw_data[index]
            img1 = self._process_image(id1)
            img2 = self._process_image(id2)
            return img1, img2, label
        else:
            return self._process_image(self.raw_data[index])

    def __len__(self):
        return len(self.raw_data)

class INatDataset(ImageDataset):
    """Dataset for iNaturalist."""
    def __init__(self, raw_data, is_pairs):
        super(INatDataset, self).__init__(raw_data, is_pairs)
        with open(inat_reader.FILENAMES_FILE) as f:
            self.id_to_filename = json.load(f)
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))

    def _process_image(self, image_id):
        path = os.path.join(inat_reader.INAT_DIR, self.id_to_filename[image_id])
        img = Image.open(path).convert('RGB')
        img = self.center_crop(img)
        img = self.tensor_aug(img)
        img = self.norm_aug(img)
        return img

class CelebADataset(ImageDataset):
    """Dataset for CelebA."""
    def __init__(self, raw_data, is_pairs):
        super(CelebADataset, self).__init__(raw_data, is_pairs)
        self.orig_min_dim = 178  # Original is 178 x 218
        self.center_crop = transforms.CenterCrop(self.orig_min_dim)
        self.resize = transforms.Resize(self.im_size)

    def _process_image(self, image_id):
        path = os.path.join(celeba_reader.CELEBA_DIR, image_id)
        img = Image.open(path).convert('RGB')
        img = self.center_crop(img)
        img = self.resize(img)
        img = self.tensor_aug(img)
        img = self.norm_aug(img)
        return img

def make_image_dataset(world_name, *args):
    if world_name == 'inat':
        return INatDataset(*args)
    elif world_name == 'celeba':
        return CelebADataset(*args)
    else:
        raise ValueError(world_name) 

class CosineLogisticBNLoss(nn.Module):
    """Like CosineLogisticLoss but with batchnorm"""
    def __init__(self, model, distance_metric='cosine', logit_scale=1e3):
        super(CosineLogisticBNLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.logit_scale = logit_scale

        self.batchnorm = nn.BatchNorm1d(1, momentum=None, eps=1e-8)  # Includes an affine transformation
        #nn.init.constant_(self.batchnorm.weight, 10 / logit_scale)  # Initialize to net weight = 10
        #nn.init.constant_(self.batchnorm.bias, -50 / logit_scale)  # Initialize to net bias = -50
        #print('Initialized bn weight', self.batchnorm.weight.detach().cpu())
        #print('Initialized bn bias', self.batchnorm.bias.detach().cpu())
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, img1, img2, labels=None, return_both=False):
        rep1 = self.model(img1)  # B, d
        rep2 = self.model(img2)  # B, d
        distance = get_distance(rep1, rep2, self.distance_metric)
        logits = self.logit_scale * self.batchnorm(distance.view(-1, 1)).view(-1) # B,
        if labels is None:
            return logits
        else:
            loss = self.loss_func(logits, labels.float())
            if return_both:
                return logits, loss
            else:
                return loss

class DistanceEmbedder(nn.Module):
    """Helper module that converts pairs of sentence features to distance.""" 
    def __init__(self, model, distance_metric='cosine'):
        super(DistanceEmbedder, self).__init__()
        self.model = model
        self.distance_metric = distance_metric

    def forward(self, img1, img2):
        rep1 = self.model(img1)  # B, d
        rep2 = self.model(img2)  # B, d
        distance = get_distance(rep1, rep2, self.distance_metric)
        return distance


class ImageEmbedder(object):
    def __init__(self, world_name, output_dir, min_epochs=1, min_train_updates=0,
                 loss_type='sgdbn', distance_metric='cosine', logit_scale=1e4,
                 dev_frac=0, batch_size=48, lr=0.001, lr_decay=0.94, 
                 epoch_decay=4, momentum=0.9, weight_decay=1e-4, 
                 dont_save_model=False):
        self.world_name = world_name
        self.output_dir = output_dir
        self.min_epochs = min_epochs
        self.min_train_updates = min_train_updates
        self.loss_type = loss_type
        self.distance_metric = distance_metric
        self.logit_scale = logit_scale
        self.dev_frac = dev_frac
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.epoch_decay = epoch_decay
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dont_save_model = dont_save_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.loss_model = None

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        if self.loss_type == 'sgdbn':
            torch.save(self.loss_model.batchnorm.state_dict(), os.path.join(path, 'batchnorm.pt'))

    def load(self, path):
        logging.info('Loading ImageEmbedder from {}'.format(path))
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        if self.loss_type == 'sgdbn':
            self.loss_model = CosineLogisticBNLoss(model=self.model, 
                                                   distance_metric=self.distance_metric, 
                                                   logit_scale=self.logit_scale)
            self.loss_model.batchnorm.load_state_dict(torch.load(os.path.join(path, 'batchnorm.pt')))
        # By default, just return cosine similarity
        self.logit_weight = 1.0
        self.logit_bias = 0.0

    def init_model(self):
        self.model = torchvision.models.resnet50(pretrained=True)

    def fit_logistic(self, dataloader, T=10000, lr_init=1e0, return_normalized=False):
        dist_emb = DistanceEmbedder(self.model, self.distance_metric)
        dist_emb.eval()
        x_features = []
        y_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Embedding in fit_logistic()'):
                img1, img2, target = batch
                img1 = img1.cuda()
                img2 = img2.cuda()
                distance = dist_emb(img1, img2)
                x_features.extend(distance.tolist())
                y_labels.extend(target.tolist())
        return fit_logistic_helper(x_features, y_labels, T=T, lr_init=lr_init, return_normalized=return_normalized)


    def _save_data(self, dataset, filename):
        # Dump collected data to file
        with open(filename, 'w') as f:
            for ex in dataset:
                id1, id2, label = ex
                print('{}\t{}\t{}'.format(id1, id2, label), file=f)

    def train(self, train_dataset, save_dir=None, dev_dataset=None, eval_on_train=True,
              keep_prev_init=False):
        if not save_dir:
            save_dir = self.output_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        num_dev = int(round(self.dev_frac * len(train_dataset)))
        if num_dev > 0:
            random.shuffle(train_dataset)
            dev_dataset = train_dataset[:num_dev]
            train_dataset = train_dataset[num_dev:]
            logging.info('Created splits: train={}, dev={}'.format(len(train_dataset), len(dev_dataset)))
        if not self.model or not keep_prev_init:
            logging.info('Initializing model.') 
            self.init_model()
        else:
            logging.info('Initializing from previous model.')

        logging.info('Preparing training data.') 
        logging.info('Stats: total={}, +={}, -={}'.format(
                len(train_dataset), 
                sum(1 for x in train_dataset if x[2] == 1), 
                sum(1 for x in train_dataset if x[2] == 0)))
        train_prep = make_image_dataset(self.world_name, train_dataset, True)
        train_loader = torch.utils.data.DataLoader(train_prep, shuffle=True, batch_size=self.batch_size,
                                                   drop_last=True, num_workers=8, pin_memory=True)
        if dev_dataset:
            raise NotImplementedError
        if self.loss_type == 'sgdbn':
            self.loss_model = CosineLogisticBNLoss(self.model, 
                                                   distance_metric=self.distance_metric, 
                                                   logit_scale=self.logit_scale)
        else:
            raise ValueError(self.loss_type)
        num_epochs = max(self.min_epochs, math.ceil(self.min_train_updates / len(train_dataset)))
        logging.info('Training for {} epoch(s).'.format(num_epochs)) 

        optimizer = torch.optim.SGD(self.loss_model.parameters(), self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        for epoch in range(num_epochs):
            _train_model(train_loader, self.loss_model, optimizer, epoch)
        if self.loss_type == 'sgdbn':
            # real_w * x + real_b = scale * ((x - mu) / (sqrt(var + eps)) * gamma + beta)
            # real_w = scale * gamma / sqrt(var + eps)
            # real_b = scale * (beta - mu * gamma / sqrt(var + eps))
            bn = self.loss_model.batchnorm
            mean_bn = bn.running_mean.detach().cpu().numpy()[0]
            var_bn = bn.running_var.detach().cpu().numpy()[0] 
            denom = np.sqrt(var_bn + bn.eps)
            w_bn = bn.weight.detach().cpu().numpy()[0]
            self.logit_weight = self.logit_scale * w_bn / denom
            self.logit_bias = self.logit_scale * (
                    bn.bias.detach().cpu().numpy() - mean_bn * w_bn / denom)
            logging.info('Final batchnorm stats: mean={}, var={}'.format(mean_bn, var_bn))
            logging.info('Final batchnorm classifier: w={}, b={}, boundary={}'.format(
                    self.logit_weight, self.logit_bias, - self.logit_bias / self.logit_weight))

        # Recalibrate!
        logging.info('Recalibrating')
        recal_w, recal_b = self.fit_logistic(train_loader)  # Find best weights on whole training dataset
        self.logit_weight = np.float64(recal_w)
        self.logit_bias = np.float64(recal_b)
        self.loss_model.eval()
        if save_dir and not self.dont_save_model:
            self.save(save_dir)
        if save_dir and num_dev > 0:
            self._save_data(train_dataset, os.path.join(save_dir, 'train_split.tsv'))
            self._save_data(dev_dataset, os.path.join(save_dir, 'dev_split.tsv'))

    def embed(self, image_ids):
        """Embed a list of image_ids using the current trained model.
        
        Args:
            questions: List[String], each string being a image ID.
        Returns:
            numpy array of size (len(questions), embedding_size)
        """
        self.model.cuda()
        self.model.eval()
        dataset = make_image_dataset(self.world_name, image_ids, False)
        # Can use 2 * self.batch_size--main batch size is for pair 
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=2 * self.batch_size,
                                             num_workers=4, pin_memory=True)
        all_output = None
        for img in tqdm(loader, desc='Embedding in ImageEmbedder'):
            img = img.cuda()
            output = self.model(img).detach().cpu().numpy()  # B, d
            if all_output is None:
                all_output = output
            else:
                all_output = np.concatenate((all_output, output))
        return all_output

    def get_logit(self, cos_sim):
        """Convert a cosine similarity into a (pre-sigmoid) logit.
        
        Should work if cos_sim is any numpy array, returns numpy array of same size.
        """
        return self.logit_weight * np.array(cos_sim) + self.logit_bias


def _train_model(train_loader, loss_model, optimizer, epoch):
    loss_model.cuda()
    loss_model.train()
    start = time.time()
    total_loss = 0.0
    num_correct = 0
    num_examples = 0
    for i, (img1, img2, target) in enumerate(tqdm(train_loader, desc='Training in ImageEmbedder')):
        end = time.time()
        img1 = img1.cuda()
        img2 = img2.cuda()
        target = target.cuda()

        # compute output
        output, loss = loss_model(img1, img2, labels=target, return_both=True)
        num_examples += target.shape[0]
        total_loss += loss.item() * target.shape[0]
        num_correct += sum(1 for pred, gold in zip(output.tolist(), target.tolist()) if (2 * gold - 1) * pred > 0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        start = time.time()
    print('Epoch loss: {}'.format(total_loss / num_examples))
    print('Epoch accuracy: {}/{} = {}%'.format(num_correct, num_examples, 100.0 * num_correct / num_examples))
