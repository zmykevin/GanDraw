# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
An inference time implementation for
recurrent GAN. Main difference is connecting
last time step generations to the next time
step. (Getting rid of Teacher Forcing) Copied from inference/reccurent_gan by Mingyang Zhou.
"""
import os
from glob import glob

import torch
import torch.nn as nn
from torch.nn import DataParallel
import cv2
import numpy as np

from geneva.models.networks.generator_factory import GeneratorFactory
from geneva.models.image_encoder import ImageEncoder
from geneva.models.sentence_encoder import SentenceEncoder
from geneva.models.condition_encoder import ConditionEncoder
from geneva.models import _recurrent_gan
from torchvision import transforms
import torchvision

LABEL2COLOR = {
            0: {"name": "sky", "color": np.array([134, 193, 46])},
            1: {"name": "dirt", "color": np.array([30, 22, 100])},
            2: {"name": "gravel", "color": np.array([163, 164, 153])},
            3: {"name": "mud", "color": np.array([35, 90, 74])},
            4: {"name": "sand", "color": np.array([196, 15, 241])},
            5: {"name": "clouds", "color": np.array([198, 182, 115])},
            6: {"name": "fog", "color": np.array([76, 60, 231])},
            7: {"name": "hill", "color": np.array([190, 128, 82])},
            8: {"name": "mountain", "color": np.array([122, 101, 17])},
            9: {"name": "river", "color": np.array([97, 140, 33])},
            10: {"name": "rock", "color": np.array([90, 90, 81])},
            11: {"name": "sea", "color": np.array([255, 252, 51])},
            12: {"name": "snow", "color": np.array([51, 255, 252])},
            13: {"name": "stone", "color": np.array([106, 107, 97])},
            14: {"name": "water", "color": np.array([0, 255, 0])},
            15: {"name": "bush", "color": np.array([204, 113, 46])},
            16: {"name": "flower", "color": np.array([0, 0, 255])},
            17: {"name": "grass", "color": np.array([255, 0, 0])},
            18: {"name": "straw", "color": np.array([255, 51, 252])},
            19: {"name": "tree", "color": np.array([255, 51, 175])},
            20: {"name": "wood", "color": np.array([66, 18, 120])},
            21: {"name": "road", "color": np.array([255, 255, 0])},
        }

LABEL2GAUGANINDEX = {
    0:156,
    1:110,
    2:124,
    3:135,
    4:14,
    5:105,
    6:119,
    7:126,
    8:134,
    9:147,
    10:149,
    11:154,
    12:158,
    13:161,
    14:177,
    15:96,
    16:118,
    17:123,
    18:162,
    19:168,
    20:181,
    21:148
}
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


class InferenceRecurrentGAN_Mingyang():

    def __init__(self, cfg):
        """A recurrent GAN model, each time step an generated image
        (x'_{t-1}) and the current question q_{t} are fed to the RNN
        to produce the conditioning vector for the GAN.
        The following equations describe this model:

            - c_{t} = RNN(h_{t-1}, q_{t}, x^{~}_{t-1})
            - x^{~}_{t} = G(z | c_{t})
        """
        super(InferenceRecurrentGAN_Mingyang, self).__init__()
        self.generator = DataParallel(
            GeneratorFactory.create_instance(cfg),
            device_ids=[0]).cuda()
        
        self.generator.eval()

        self.rnn = nn.DataParallel(
            nn.GRU(cfg.input_dim, cfg.hidden_dim,
                   batch_first=False),
            dim=1,
            device_ids=[0]).cuda()
        self.rnn.eval()

        self.layer_norm = nn.DataParallel(nn.LayerNorm(cfg.hidden_dim),
                                          device_ids=[0]).cuda()
        self.layer_norm.eval()

        self.image_encoder = DataParallel(ImageEncoder(cfg),
                                          device_ids=[0]).cuda()
        self.image_encoder.eval()

        self.condition_encoder = DataParallel(ConditionEncoder(cfg),
                                              device_ids=[0]).cuda()
        self.condition_encoder.eval()

        self.sentence_encoder = nn.DataParallel(SentenceEncoder(cfg),
                                                device_ids=[0]).cuda()
        self.sentence_encoder.eval()

        self.cfg = cfg
        self.results_path = cfg.results_path
        if not os.path.exists(cfg.results_path):
            os.mkdir(cfg.results_path)
        self.unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(
                                 0.5, 0.5, 0.5))
        #self.reset_drawer()

    def predict(self, batch, iteration=0, visualize_batch=0, visualize_progress=False, visualize_images=[], visualizer=None):
        with torch.no_grad():
            batch_size = len(batch['image'])
            #print("evaluation batch size is: {}".format(batch_size))
            max_seq_len = batch['image'].size(1)
            scene_id = batch['scene_id']
            #print(scene_id[0])


            # print(dialog_l)
            # Initial inputs for the RNN set to zeros
            prev_image = torch.FloatTensor(batch['background'])\
                .repeat(batch_size, 1, 1, 1)
            # print("The previous image size is from background")
            # print(prev_image.shape)
            hidden = torch.zeros(1, batch_size, self.cfg.hidden_dim)
            generated_images = []
            gt_images = []

            target_image = batch['target_image']
            image_gen_mode = self.cfg.image_gen_mode
            #print("target_image shape is: {}".format(target_image.shape))
            if visualize_progress:
                total_visualization = 5 if 5 < max_seq_len else max_seq_len
                # Put the ground truth of images
                if not visualize_images:
                    # visualize_images.extend(
                    #     [target_image[0, x] for x in range(total_visualization)])
                    for x in range(total_visualization):
                        current_target = batch["image"][visualize_batch, x]
                        # process the image
                        if image_gen_mode == "real":
                            current_target = self.unorm(current_target)
                            current_target = transforms.ToPILImage()(current_target).convert('RGB')
                            # new_x = np.array(new_x)[..., ::-1]
                            current_target = np.moveaxis(np.array(current_target), -1, 0)
                        elif image_gen_mode == "segmentation_onehot":
                            current_target_raw = current_target.data.cpu()
                            seg_map = np.argmax(current_target_raw, axis=0)
                            current_target = np.zeros((3, seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
                            for i in range(seg_map.shape[0]):
                                for j in range(seg_map.shape[1]):
                                    #print(seg_map[i,j].item())
                                    current_target[:,i,j] = LABEL2COLOR[seg_map[i,j].item()]["color"]

                        visualize_images.append(current_target)


            for t in range(max_seq_len):
                turns_word_embedding = batch['turn_word_embedding'][:, t]
                turns_lengths = batch['turn_lengths'][:, t]

                image_feature_map, image_vec, object_detections = self.image_encoder(
                    prev_image)

                turn_embedding = self.sentence_encoder(
                    turns_word_embedding, turns_lengths)
                rnn_condition, _ = self.condition_encoder(turn_embedding,
                                                          image_vec)

                rnn_condition = rnn_condition.unsqueeze(0)
                output, hidden = self.rnn(rnn_condition,
                                          hidden)

                output = output.squeeze(0)
                output = self.layer_norm(output)

               #print("image feature map size is: {}".format(image_feature_map.shape))
                generated_image = self._forward_generator(batch_size, output,
                                                          image_feature_map)

                if (not self.cfg.inference_save_last_only) or (self.cfg.inference_save_last_only and t == max_seq_len - 1):
                    generated_images.append(generated_image)
                    gt_images.append(batch['image'][:, t])
                prev_image = generated_image

                if visualize_progress:
                    if t < total_visualization:
                        current_generated_im = generated_image[visualize_batch]
                        # process the image
                        if image_gen_mode == "real":
                            current_generated_im = self.unorm(
                                current_generated_im.data.cpu())
                            current_generated_im = transforms.ToPILImage()(
                                current_generated_im).convert('RGB')
                            current_generated_im = np.moveaxis(
                                np.array(current_generated_im), -1, 0)
                        elif image_gen_mode == "segmentation_onehot":
                            current_generated_im_raw = current_generated_im.data.cpu()
                            seg_map = np.argmax(current_generated_im_raw, axis=0)
                            current_generated_im = np.zeros((3, seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
                            for i in range(seg_map.shape[0]):
                                for j in range(seg_map.shape[1]):
                                    current_generated_im[:,i,j] = LABEL2COLOR[seg_map[i,j].item()]["color"]
                        visualize_images.append(current_generated_im)

            if visualize_progress:
                # Call the function to visualize the images
                #n_row = len(visualize_images) // total_visualization
                n_row = total_visualization
                #print("Draw n rows: {}".format(n_row))
                self._draw_images(visualizer, visualize_images, n_row)

        _save_predictions(generated_images, batch['turn'], scene_id, self.results_path, gt_images, unorm=self.unorm, target_im=target_image, iteration=iteration, image_gen_mode=self.cfg.image_gen_mode)

    def _forward_generator(self, batch_size, condition, image_feature_maps):
        noise = torch.FloatTensor(batch_size,
                                  self.cfg.noise_dim).normal_(0, 1).cuda()

        fake_images, _, _, _ = self.generator(
            noise, condition, image_feature_maps)

        return fake_images

    def load(self, pre_trained_path, iteration=None):
        snapshot = _read_weights(pre_trained_path, iteration)

        self.generator.load_state_dict(snapshot['generator_state_dict'])
        self.rnn.load_state_dict(snapshot['rnn_state_dict'])
        self.layer_norm.load_state_dict(snapshot['layer_norm_state_dict'])
        self.image_encoder.load_state_dict(
            snapshot['image_encoder_state_dict'])
        self.condition_encoder.load_state_dict(
            snapshot['condition_encoder_state_dict'])
        self.sentence_encoder.load_state_dict(
            snapshot['sentence_encoder_state_dict'])

    def _draw_images(self, visualizer, visualize_images, nrow):
        _recurrent_gan.draw_images_gandraw_visualization(
            self, visualizer, visualize_images, nrow)  # Changed by Mingyang Zhou
    def generate_im(self, current_turns_embedding, current_turns_lengths, batch_size=1):
        with torch.no_grad():
            #define batch_size
            image_feature_map, image_vec, object_detections = self.image_encoder(self.drawer_prev_img)

            turn_embedding = self.sentence_encoder(current_turns_embedding, current_turns_lengths)
            rnn_condition, _ = self.condition_encoder(turn_embedding,image_vec)

            rnn_condition = rnn_condition.unsqueeze(0)
            output, hidden = self.rnn(rnn_condition,
                                      self.drawer_hidden)

            output = output.squeeze(0)
            output = self.layer_norm(output)

           #print("image feature map size is: {}".format(image_feature_map.shape))
            generated_image = self._forward_generator(batch_size, output,
                                                      image_feature_map)
            #update prev_img

            self.drawer_prev_img = generated_image
            self.drawer_hidden = hidden

            #Convert generated_image to format can be taken by GanDraw
            generated_image_raw = generated_image[0].data.cpu()
            generated_seg_map = np.argmax(generated_image_raw, axis=0)
            #Convert Seg_Map to the Format that can ba handeled by GanDraw
            output_img = np.zeros((generated_seg_map.shape[0], generated_seg_map.shape[1]),dtype=np.int32)
            output_img_colorful = np.zeros((3, generated_seg_map.shape[0], generated_seg_map.shape[1]), dtype=np.uint8)
            for i in range(generated_seg_map.shape[0]):
                for j in range(generated_seg_map.shape[1]):
                    output_img_colorful[:,i,j] = LABEL2COLOR[generated_seg_map[i,j].item()]["color"]
                    #print(generated_seg_map[i,j].item())
                    output_img[i,j] = LABEL2GAUGANINDEX[generated_seg_map[i,j].item()]
            output_img = np.uint8(output_img)
        return output_img


    def reset_drawer(self,background_img,batch_size=1):
        #reinitialize_drawer
        self.drawer_prev_img = torch.FloatTensor(background_img).repeat(batch_size, 1, 1, 1) #To repeat the process
        self.drawer_hidden = torch.zeros(1, batch_size, self.cfg.hidden_dim)
   






def unormalize(x, unorm):
    """
    unormalize the image
    """
    new_x = unorm(x)
    new_x = transforms.ToPILImage()(new_x).convert('RGB')
    new_x = np.array(new_x)[..., ::-1]
    #new_x = np.moveaxis(np.array(new_x), -1, 0)
    return new_x

def unormalize_segmentation(x):
    new_x = (x + 1) * 128
    new_x = new_x.transpose(1, 2, 0)[..., ::-1]
    return new_x

def unormalize_segmentation_onehot(x):
        """
        Convert the segmentation into image
        """

        # LABEL2COLOR = {
        #     0: {"name": "sky", "color": np.array([134, 193, 46])},
        #     1: {"name": "dirt", "color": np.array([30, 22, 100])},
        #     2: {"name": "gravel", "color": np.array([163, 164, 153])},
        #     3: {"name": "mud", "color": np.array([35, 90, 74])},
        #     4: {"name": "sand", "color": np.array([196, 15, 241])},
        #     5: {"name": "clouds", "color": np.array([198, 182, 115])},
        #     6: {"name": "fog", "color": np.array([76, 60, 231])},
        #     7: {"name": "hill", "color": np.array([190, 128, 82])},
        #     8: {"name": "mountain", "color": np.array([122, 101, 17])},
        #     9: {"name": "river", "color": np.array([97, 140, 33])},
        #     10: {"name": "rock", "color": np.array([90, 90, 81])},
        #     11: {"name": "sea", "color": np.array([255, 252, 51])},
        #     12: {"name": "snow", "color": np.array([51, 255, 252])},
        #     13: {"name": "stone", "color": np.array([106, 107, 97])},
        #     14: {"name": "water", "color": np.array([0, 255, 0])},
        #     15: {"name": "bush", "color": np.array([204, 113, 46])},
        #     16: {"name": "flower", "color": np.array([0, 0, 255])},
        #     17: {"name": "grass", "color": np.array([255, 0, 0])},
        #     18: {"name": "straw", "color": np.array([255, 51, 252])},
        #     19: {"name": "tree", "color": np.array([255, 51, 175])},
        #     20: {"name": "wood", "color": np.array([66, 18, 120])},
        #     21: {"name": "road", "color": np.array([255, 255, 0])},
        # }
        seg_map = np.argmax(x, axis=0)
        new_x = np.zeros((3, seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
        for i in range(seg_map.shape[0]):
            for j in range(seg_map.shape[1]):
                new_x[:,i,j] = LABEL2COLOR[seg_map[i,j].item()]["color"]
        new_x = new_x.transpose(1,2,0)[..., ::-1]
        return new_x

def _save_predictions(images, text, scene_id, results_path, gt_images, unorm=None, target_im=None, iteration=0, image_gen_mode="real"):
    for i, scene in enumerate(scene_id):
        #check the scene
        if not os.path.exists(os.path.join(results_path, str(scene))):
            os.mkdir(os.path.join(results_path, str(scene)))
        if not os.path.exists(os.path.join(results_path, str(scene) + '_gt')):
            os.mkdir(os.path.join(results_path, str(scene) + '_gt'))

        for t in range(len(images)):
            if t >= len(text[i]):
                continue
            # image = (images[t][i].data.cpu().numpy() + 1) * 128
            # image = image.transpose(1, 2, 0)[..., ::-1]
            # image = unorm(images[t][i].data.cpu())
            # image = transforms.ToPILImage()(image).convert('RGB')
            # image = np.array(image)[..., ::-1]
            if image_gen_mode == "real":
                image = unormalize(images[t][i].data.cpu(), unorm)
            elif image_gen_mode == "segmentation":
                image = unormalize_segmentation(images[t][i].data.cpu())
            elif image_gen_mode == "segmentation_onehot":
                image = unormalize_segmentation_onehot(images[t][i].data.cpu())
            #query = text[i][t]
            #query = '_'.join(query.split())
            # gt_image = (gt_images[t][i].data.cpu().numpy() + 1) * 128
            # gt_image = gt_image.transpose(1, 2, 0)[..., ::-1]
            if image_gen_mode in ["real", "segmentation"]:
                gt_image = unorm(gt_images[t][i].data.cpu())
                gt_image = transforms.ToPILImage()(gt_image).convert('RGB')
                gt_image = np.array(gt_image)[..., ::-1]
            elif image_gen_mode == "segmentation_onehot":
                gt_image = unormalize_segmentation_onehot(gt_images[t][i].data.cpu())
            
            #print(gt_image)
            #print(os.path.join(results_path, str(scene) + '_gt', '{}_{}.png'.format(t, query)))
            #cv2.imwrite(os.path.join(results_path, str(scene), '{}_{}_{}.png'.format(t, query, iteration)),image)
            cv2.imwrite(os.path.join(results_path, str(scene), '{}_{}.png'.format(t, iteration)),image)
            #cv2.imwrite(os.path.join(results_path, str(scene) + '_gt', '{}_{}.png'.format(t, query)),gt_image)
            cv2.imwrite(os.path.join(results_path, str(scene) + '_gt', '{}.png'.format(t)), gt_image)

            # cv2.imwrite(os.path.join(results_path, str(scene), '{}.png'.format(t)),
            #             image)
            # cv2.imwrite(os.path.join(results_path, str(scene) + '_gt',
            # '{}.png'.format(t)),gt_image)
        
        # target_image = unorm(target_im[i][0].data.cpu())
        # target_image = transforms.ToPILImage()(target_image).convert('RGB')
        # target_image = np.array(target_image)[..., ::-1]
        if image_gen_mode == "real":
            target_image = unormalize(target_im[i][0].data.cpu(), unorm)
        elif image_gen_mode == "segmentation":
            target_image = unormalize_segmentation(target_im[i][0].data.cpu())
        elif image_gen_mode == "segmentation_onehot":
            target_image = unormalize_segmentation_onehot(target_im[i][0].data.cpu())
        
        cv2.imwrite(os.path.join(results_path, str(scene) + '_gt', 'target.png'),
                    target_image)


def _read_weights(pre_trained_path, iteration):
    if iteration is None:
        iteration = ''
    iteration = str(iteration)
    try:
        snapshot = torch.load(
            glob('{}/snapshot_{}*'.format(pre_trained_path, iteration))[0])
    except IndexError:
        snapshot = torch.load(pre_trained_path)
    return snapshot
