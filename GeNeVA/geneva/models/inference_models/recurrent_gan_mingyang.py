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

        self.rnn = nn.DataParallel(
            nn.GRU(cfg.input_dim, cfg.hidden_dim,
                   batch_first=False),
            dim=1,
            device_ids=[0]).cuda()

        self.layer_norm = nn.DataParallel(nn.LayerNorm(cfg.hidden_dim),
                                          device_ids=[0]).cuda()

        self.image_encoder = DataParallel(ImageEncoder(cfg),
                                          device_ids=[0]).cuda()

        self.condition_encoder = DataParallel(ConditionEncoder(cfg),
                                              device_ids=[0]).cuda()

        self.sentence_encoder = nn.DataParallel(SentenceEncoder(cfg),
                                                device_ids=[0]).cuda()

        self.cfg = cfg
        self.results_path = cfg.results_path
        if not os.path.exists(cfg.results_path):
            os.mkdir(cfg.results_path)
        self.unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(
                                 0.5, 0.5, 0.5))

    def predict(self, batch, iteration=0, visualize_batch=0, visualize_progress=False, visualize_images=[], visualizer=None):
        with torch.no_grad():
            batch_size = len(batch['image'])
            #print("evaluation batch size is: {}".format(batch_size))
            max_seq_len = batch['image'].size(1)
            scene_id = batch['scene_id']

            # print(dialog_l)
            # Initial inputs for the RNN set to zeros
            prev_image = torch.FloatTensor(batch['background'])\
                .repeat(batch_size, 1, 1, 1)
            # print("The previous image size is from background")
            # print(prev_image.shape)
            hidden = torch.zeros(1, batch_size, self.cfg.hidden_dim)
            generated_images = []
            gt_images = []

            target_image = batch['image']
            if visualize_progress:
                total_visualization = 5 if 5 < max_seq_len else max_seq_len
                # Put the ground truth of images
                if not visualize_images:
                    # visualize_images.extend(
                    #     [target_image[0, x] for x in range(total_visualization)])
                    for x in range(total_visualization):
                        current_target = target_image[visualize_batch, x]
                        # process the image
                        current_target = self.unorm(current_target)
                        current_target = transforms.ToPILImage()(current_target).convert('RGB')
                        # new_x = np.array(new_x)[..., ::-1]
                        current_target = np.moveaxis(
                            np.array(current_target), -1, 0)
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
                        current_generated_im = self.unorm(
                            current_generated_im.data.cpu())
                        current_generated_im = transforms.ToPILImage()(
                            current_generated_im).convert('RGB')
                        current_generated_im = np.moveaxis(
                            np.array(current_generated_im), -1, 0)
                        visualize_images.append(current_generated_im)

            if visualize_progress:
                # Call the function to visualize the images
                #n_row = len(visualize_images) // total_visualization
                n_row = total_visualization
                #print("Draw n rows: {}".format(n_row))
                self._draw_images(visualizer, visualize_images, n_row)

        _save_predictions(generated_images, batch[
                          'turn'], scene_id, self.results_path, gt_images, unorm=self.unorm, target_im=target_image, iteration=iteration)

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


def _save_predictions(images, text, scene_id, results_path, gt_images, unorm=None, target_im=None, iteration=0):
    for i, scene in enumerate(scene_id):
        if not os.path.exists(os.path.join(results_path, str(scene))):
            os.mkdir(os.path.join(results_path, str(scene)))
        if not os.path.exists(os.path.join(results_path, str(scene) + '_gt')):
            os.mkdir(os.path.join(results_path, str(scene) + '_gt'))
        for t in range(len(images)):
            if t >= len(text[i]):
                continue
            # image = (images[t][i].data.cpu().numpy() + 1) * 128
            # image = image.transpose(1, 2, 0)[..., ::-1]
            image = unorm(images[t][i].data.cpu())
            image = transforms.ToPILImage()(image).convert('RGB')
            image = np.array(image)[..., ::-1]

            query = text[i][t]
            # gt_image = (gt_images[t][i].data.cpu().numpy() + 1) * 128
            # gt_image = gt_image.transpose(1, 2, 0)[..., ::-1]
            gt_image = unorm(gt_images[t][i].data.cpu())
            gt_image = transforms.ToPILImage()(gt_image).convert('RGB')
            gt_image = np.array(gt_image)[..., ::-1]

            cv2.imwrite(os.path.join(results_path, str(scene), '{}_{}_{}.png'.format(t, query, iteration)),
                        image)
            cv2.imwrite(os.path.join(results_path, str(scene) + '_gt', '{}_{}.png'.format(t, query)),
                        gt_image)

            # cv2.imwrite(os.path.join(results_path, str(scene), '{}.png'.format(t)),
            #             image)
            # cv2.imwrite(os.path.join(results_path, str(scene) + '_gt',
            # '{}.png'.format(t)),gt_image)
        target_image = unorm(target_im[i][0].data.cpu())
        target_image = transforms.ToPILImage()(target_image).convert('RGB')
        target_image = np.array(target_image)[..., ::-1]
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
