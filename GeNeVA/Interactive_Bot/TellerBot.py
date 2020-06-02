import os
import glob
import cv2

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json  # Added to initialize the setting in Jupyter Notebook-by Mingyang
import easydict  # Added to initialize the setting in Jupyter Notebook-by Mingyang
import random
import numpy as np
import nltk
import string
from PIL import Image
from pathlib import Path

from geneva.models.models import INFERENCE_MODELS
from geneva.data.datasets import DATASETS
from geneva.evaluation.evaluate import Evaluator
from geneva.utils.config import keys, parse_config
from geneva.utils.visualize import VisdomPlotter
from geneva.models.models import MODELS
from geneva.data import codraw_dataset
from geneva.data import clevr_dataset
from geneva.data import gandraw_dataset
from geneva.evaluation.seg_scene_similarity_score import report_gandraw_eval_result
from geneva.utils.config import keys
from math import sqrt

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def compute_mean(self):
        self.avg = self.sum / self.count
        
class GanDraw_Teller():
    def __init__(self, cfg, pretrained_model_path=None, iteration=6000):
        self.cfg = cfg
        self.cfg.batch_size = 1
        #Load the Dataset
        dataset_path = cfg.test_dataset
        gandraw_vocab_path = "/home/zmykevin/CoDraw_Gaugan/data/GanDraw/data_full/gandraw_vocab.txt"
        with open(gandraw_vocab_path, 'r') as f:
            gandraw_vocab = f.readlines()
            gandraw_vocab = [x.strip().rsplit(' ', 1)[0] for x in gandraw_vocab]        
        self.vocab = ['<s_start>', '<s_end>', '<unk>', '<pad>', '<d_end>'] + gandraw_vocab        
        self.cfg.vocab_size = len(self.vocab)
        #print(self.cfg.vocab_size)
        #self.cfg.vocab_size = self.dataset.vocab_size
        self.model = INFERENCE_MODELS[cfg.gan_type](cfg)
        #load the pretrained_model
        if pretrained_model_path is not None:
            self.model.load_model('/'.join([pretrained_model_path,'snapshot_{}.pth'.format(iteration)]))
            
        #self.iterations = len(self.dataset) // cfg.batch_size
        self.current_iteration=iteration
        #define the collate_fn
        #self.dataloader.collate_fn = gandraw_dataset.collate_data
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.output_dir = self.cfg.results_path
        
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        self.unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(
                                 0.5, 0.5, 0.5))
        #self.unk_embedding = np.load("unk_embedding.npy")
        self.word2index = {k: v for v, k in enumerate(self.vocab)}
        self.index2word = {v: k for v, k in enumerate(self.vocab)}
        self.previous_output_utt = None
        self.reset_teller()
    def generate_utt(self, input_img=None, input_utt=None):
        #concate output_utt and  input_utt
        if input_img is not None and input_utt is not None:
            input_img = self.load_img(input_img)
            input_utt_ids, input_utt_len = self.utt2ids(input_utt)
        else:
            input_utt_ids = None
            input_utt_len = None
        output_utt, stop =  self.model.generate_utt(input_img,input_utt_ids, input_utt_len, self.word2index, self.index2word)
        #update self.previous_output_utt
        self.previous_output_utt = output_utt
        return  output_utt, stop
    def load_img(self, input_img, resize_wh=128):
        r"""
        input_img should be a three dimensional numpy matrix
        """
        if input_img.shape[0] > resize_wh:
            input_img = cv2.resize(input_img, (resize_wh, resize_wh), interpolation=cv2.INTER_AREA)
        
        processed_input_img = self.image_transform(input_img).numpy()
        processed_input_img = np.expand_dims(processed_input_img, axis=0)
        return torch.FloatTensor(processed_input_img)
    def utt2ids(self, input_text):
        #Tokenize the input_text
        teller_text_tokens = ['<teller>'] + nltk.word_tokenize(self.previous_output_utt)
        drawer_text_tokens = ['<drawer>'] + nltk.word_tokenize(input_text)
        all_tokens = teller_text_tokens + drawer_text_tokens
        all_tokens_ids = [0] + [self.word2index.get(x, self.word2index['<unk>']) for x in all_tokens if x!= "<teller>" and x!= "<drawer>"]+[1]
        all_tokens_len = len(all_tokens_ids)
        
        turn_teller_drawer_ids = np.array(all_tokens_ids)
        turn_teller_drawer_ids = np.expand_dims(turn_teller_drawer_ids, axis=0)
        turn_teller_drawer_ids_len = np.ones((1))*all_tokens_len
        
        return torch.LongTensor(turn_teller_drawer_ids), torch.LongTensor(turn_teller_drawer_ids_len)
    def reset_teller(self):
        self.previous_output_utt = None
        self.model.reset_teller()
    def import_tgt_img(self, tgt_img):
        tgt_image = self.load_img(tgt_img)
        self.model.import_tgt_img(tgt_image)