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

def _parse_glove(glove_path):
    glove = {}
    with open(glove_path, 'r') as f:
        for line in f:
            splitline = line.split()
            word = splitline[0]
            embedding = np.array([float(val) for val in splitline[1:]])
            glove[word] = embedding

    return glove

class GanDraw_Silent_Drawer():
    def __init__(self, cfg, pretrained_model_path=None, iteration=1000):
        self.cfg = cfg
        self.model = INFERENCE_MODELS[cfg.gan_type](cfg)
        #load the pretrained_model
        if pretrained_model_path is not None:
            self.model.load(pretrained_model_path,iteration)
        #Load the Dataset
        dataset_path = cfg.test_dataset
        self.dataset = DATASETS[cfg.dataset](path=keys[dataset_path],
                                             cfg=cfg,
                                             img_size=cfg.img_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=cfg.batch_size,
                                     shuffle=False,
                                     num_workers=cfg.num_workers,
                                     drop_last=True)
        self.iterations = len(self.dataset) // cfg.batch_size
        self.current_iteration=iteration
        #define the collate_fn
        self.dataloader.collate_fn = gandraw_dataset.collate_data
        
        
        self.visualize_batch = 0
        # Keep all the progress images to be processed.
        self.visualize_images = []
        
        self.default_drawer_utt = ["okay", "done", "next"]
        self.glove = _parse_glove(keys['glove_gandraw_path'])
        self.unk_embedding = np.load("unk_embedding.npy")
        self.get_background_embedding()
        
    def get_background_embedding(self):
        self.background_embedding = np.zeros((self.cfg.img_size, self.cfg.img_size, 22), dtype=np.int32)
        self.background_embedding[:,:,0] = 1 #Define the background with sky label activated
        self.background_embedding =  np.expand_dims(self.background_embedding, axis=0)
        self.background_embedding = self.process_image(self.background_embedding)
        self.background_embedding = torch.FloatTensor(self.background_embedding)
    
    def run_test(self,visualize_progress=False):
        for batch in tqdm(self.dataloader, total=self.iterations):
            self.model.predict(
                batch, iteration=self.current_iteration, visualize_batch=self.visualize_batch, visualize_progress=visualize_progress, visualize_images=self.visualize_images, visualizer=None)
        #run the evaluation metrics
        metrics_report = dict()
        scene_sim_score, meanIoU, precision, recall, acc, F1 = report_gandraw_eval_result(None, self.current_iteration, self.cfg.results_path, use_test=True)
        metrics_report['scene_sim_score'] =  scene_sim_score
        metrics_report['meanIoU'] = meanIoU
        metrics_report['precision'] = precision
        metrics_report['recall'] = recall
        metrics_report['acc'] = acc
        metrics_report['F1'] = F1
        
        return metrics_report
    def generate_im(self, input_text):
        #TODO: build the function to generate_im
        with torch.no_grad():
            current_turn_embedding, current_turn_len = self.utt2embedding(input_text)
            gen_im = self.model.generate_im(current_turn_embedding, current_turn_len)
            gen_im = self.post_processing_im(gen_im)
        return gen_im
    def utt2embedding(self, input_text):
        #Tokenize the input_text
        text_tokens = ['<teller>'] + nltk.word_tokenize(input_text)
        sampled_drawer_utt = ['<drawer>']+nltk.word_tokenize(random.choice(self.default_drawer_utt))
        text_tokens = text_tokens + sampled_drawer_utt
        #get padded_input_text
        processed_text_tokens =  [w for w in text_tokens if w not in string.punctuation]
        processed_text_len = len(processed_text_tokens)
        #initialize turn embedding 
        turn_embeddings = np.zeros((processed_text_len, 300))
        for i,w in enumerate(processed_text_tokens):
            turn_embeddings[i] = self.glove.get(w, self.unk_embedding)
        #turns_embeddings is not a numpy matrix
        turn_embeddings = np.expand_dims(turn_embeddings, axis=0)
        turn_lens = np.ones((1))*processed_text_len
        
        return torch.FloatTensor(turn_embeddings), torch.LongTensor(turn_lens)
    def reset_drawer(self):
        self.model.reset_drawer(self.background_embedding)
        #self.model.eval()
    def post_processing_im(self, gen_im,resize_wh=512):
        dominant_label = np.unique(gen_im)
        output_image = cv2.resize(gen_im, (resize_wh, resize_wh), interpolation=cv2.INTER_AREA)
        output_image = self.smooth_segmentation(output_image, dominant_label)
        return output_image
    def smooth_segmentation(self, image, dominant_label):
        """
        image is the 3D gray scale image with each pixel equal to the label of a certain category.
        return the same size of shrinked_image with only dominant_label
        """
        drawing2landscape = [
            ([0, 0, 0],156), #sky
            ([156, 156, 156], 156),#sky
            ([154, 154, 154], 154), #sea
            ([134, 134, 134], 134), #mountain
            ([149, 149, 149], 149), #rock
            ([126, 126, 126], 126), #hill
            ([105, 105, 105], 105), #clouds
            ([14, 14, 14], 14), #sand
            ([124, 124, 124], 124), #gravel
            ([158, 158, 158], 158), #snow
            ([147, 147, 147], 147), #river
            ([96, 96, 96], 96), #bush
            ([168, 168, 168], 168), #tree
            ([148, 148, 148], 148), #road
            ([110, 110, 110], 110), #dirt 
            ([135, 135, 135], 135), #mud 
            ([119, 119, 119], 119), #fog 
            ([161, 161, 161], 161), #stone
            ([177, 177, 177], 177), #water
            ([118, 118, 118], 118), #flower
            ([123, 123, 123], 123), #grass
            ([162, 162, 162], 162), #straw
        ]

        center = []
        for l in dominant_label:
            center_array = np.array([l]*3)
            center.append(np.uint8(center_array))
        #print(center)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                current_pixel = np.uint8(image[i,j])
                #sort centers

                if not any(all(current_pixel == x) for x in center):
                    #print("sort_center")
                    center.sort(key=lambda c: sqrt((current_pixel[0]-c[0])**2+(current_pixel[1]-c[1])**2+(current_pixel[2]-c[2])**2))
                    image[i,j] = center[0]
        #print(image)
        return image
    def process_image(self, images):
        if self.cfg.image_gen_mode == "real":
            result_images = np.zeros_like(
                images.transpose(0, 3, 1, 2), dtype=np.float32)
            for i in range(images.shape[0]):
                current_img = images[i]
                current_processed_img = self.image_transform(current_img)
                current_processed_img = current_processed_img.numpy()
                result_images[i] = current_processed_img
        
        elif self.cfg.image_gen_mode == "segmentation":
            result_images = images[..., ::-1]
            #print(result_images.shape)
            result_images = result_images / 128. - 1
            result_images += np.random.uniform(size=result_images.shape, low=0, high=1. / 64)
            result_images = result_images.transpose(0, 3, 1, 2)
        elif self.cfg.image_gen_mode == "segmentation_onehot":
            #We don't preprocess the image in this setting, switch the channel to the second dimension
            result_images = images.transpose(0,3,1,2)
        return result_images
        