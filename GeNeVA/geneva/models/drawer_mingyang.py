# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
A recurrent GAN model that draws images
based on dialog/description turns in sequence. Copied from reccurent_gan.py by Mingyang Zhou
"""
import os
import gc

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel  # Mingyang Zhou
import numpy as np
import datetime  # Added by Mingyang Zhou

from geneva.models.networks.generator_factory import GeneratorFactory
from geneva.models.networks.discriminator_factory import DiscriminatorFactory
from geneva.criticism.losses import LOSSES
from geneva.models.image_encoder import ImageEncoder
from geneva.models.sentence_encoder import SentenceEncoder
from geneva.models.condition_encoder import ConditionEncoder
from geneva.models.drawer_image_encoder import DrawerImageEncoder
from geneva.models.teller_dialog_encoder import TellerDialogEncoder
from geneva.models.drawer_dialog_decoder import DrawerDialogDecoder
from geneva.inference.optim import OPTIM
from geneva.definitions.regularizers import gradient_penalty, kl_penalty
from geneva.utils.logger import Logger
from geneva.models import _recurrent_gan
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


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


class Drawer():

    def __init__(self, cfg):
        """A recurrent GAN model, each time step a generated image
        (x'_{t-1}) and the current question q_{t} are fed to the RNN
        to produce the conditioning vector for the GAN.
        The following equations describe this model:

            - c_{t} = RNN(h_{t-1}, q_{t}, x^{~}_{t-1})
            - x^{~}_{t} = G(z | c_{t})
        """
        super(Drawer, self).__init__()

        # region Models-Instantiation

        ###############################Original DataParallel###################
        self.generator = DataParallel(
            GeneratorFactory.create_instance(cfg)).cuda()

        self.discriminator = DataParallel(
            DiscriminatorFactory.create_instance(cfg)).cuda()

        self.rnn = nn.DataParallel(nn.GRU(cfg.input_dim,
                                          cfg.hidden_dim,
                                          batch_first=False), dim=1).cuda()
        # self.rnn = DistributedDataParallel(nn.GRU(cfg.input_dim,
        #                                           cfg.hidden_dim,
        # batch_first=False), dim=1).cuda()

        self.layer_norm = nn.DataParallel(nn.LayerNorm(cfg.hidden_dim)).cuda()

        self.image_encoder = DataParallel(ImageEncoder(cfg)).cuda()

        self.condition_encoder = DataParallel(ConditionEncoder(cfg)).cuda()

        self.sentence_encoder = nn.DataParallel(SentenceEncoder(cfg)).cuda()
        
        #############################Utterance Generation Module################
        # Drawer Img Encoder
        self.drawer_img_encoder = DataParallel(DrawerImageEncoder(cfg=cfg)).cuda()
        #self.img_encoder = TellerImageEncoder(network=cfg.teller_img_encoder_net)

        # 2. Dialog Encoder
        self.dialog_encoder = DataParallel(TellerDialogEncoder(cfg=cfg)).cuda()

        # 2. Caption Decoder
        self.utterance_decoder = DataParallel(DrawerDialogDecoder(cfg=cfg)).cuda()

        ########################################################################

        self.generator_optimizer = OPTIM[cfg.generator_optimizer](
            self.generator.parameters(),
            cfg.generator_lr,
            cfg.generator_beta1,
            cfg.generator_beta2,
            cfg.generator_weight_decay)

        self.discriminator_optimizer = OPTIM[cfg.discriminator_optimizer](
            self.discriminator.parameters(),
            cfg.discriminator_lr,
            cfg.discriminator_beta1,
            cfg.discriminator_beta2,
            cfg.discriminator_weight_decay)

        self.rnn_optimizer = OPTIM[cfg.rnn_optimizer](
            self.rnn.parameters(),
            cfg.rnn_lr)

        self.sentence_encoder_optimizer = OPTIM[cfg.gru_optimizer](
            self.sentence_encoder.parameters(),
            cfg.gru_lr)
        
        parameters = []
        # Initialize the optimizer for dialog
        parameters.extend(self.drawer_img_encoder.parameters())
        parameters.extend(self.utterance_decoder.parameters())
        parameters.extend(self.dialog_encoder.parameters())
        
        self.dialog_optimizer = optim.Adam(parameters, lr=cfg.teller_lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.dialog_optimizer,  cfg.teller_step_size)

        self.use_image_encoder = cfg.use_fg
        feature_encoding_params = list(self.condition_encoder.parameters())
        if self.use_image_encoder:
            feature_encoding_params += list(self.image_encoder.parameters())

        self.feature_encoders_optimizer = OPTIM['adam'](
            feature_encoding_params,
            cfg.feature_encoder_lr
        )

        # endregion

        # region Criterion

        self.criterion = LOSSES[cfg.criterion]()
        self.aux_criterion = DataParallel(torch.nn.BCELoss()).cuda()
        #Define the optimization for dialog_criterion
        self.dialog_criterion = DataParallel(torch.nn.CrossEntropyLoss()).cuda()

        #Added by Mingyang for segmentation loss 
        if cfg.balanced_seg:
            label_weights = np.array([3.02674201e-01, 1.91545454e-03, 2.90009221e-04, 7.50949673e-04, 
                                      1.08670452e-03, 1.11353785e-01, 4.00971053e-04, 1.06240113e-02,
                                      1.59590824e-01, 5.38960105e-02, 3.36431602e-02, 3.99029734e-02,
                                      1.88888847e-02, 2.06441476e-03, 6.33775290e-02, 5.81920411e-03,
                                      3.79528817e-03, 7.87975754e-02, 2.73547355e-03, 1.08308135e-01,
                                      0.00000000e+00, 8.44408475e-05])
            #reverse the loss
            label_weights = 1/label_weights
            label_weights[20] = 0
            label_weights = label_weights/np.min(label_weights[:20])
            #convert numpy to tensor
            label_weights = torch.from_numpy(label_weights)
            label_weights = label_weights.type(torch.FloatTensor)
            self.seg_criterion = DataParallel(torch.nn.CrossEntropyLoss(weight=label_weights)).cuda()
        else:
            self.seg_criterion = DataParallel(torch.nn.CrossEntropyLoss()).cuda()

        # define unorm
        self.unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(
                                 0.5, 0.5, 0.5))

        # endregion

        self.cfg = cfg
        self.logger = Logger(cfg.log_path, cfg.exp_name)

    def train_batch(self, batch, epoch, iteration, visualizer, logger, total_iters=0, current_batch_t=0):
        """
        The training scheme follows the following:
            - Discriminator and Generator is updated every time step.
            - RNN, SentenceEncoder and ImageEncoder parameters are
            updated every sequence
        """
        batch_size = len(batch['image'])
        max_seq_len = batch['image'].size(1)
        drawer_turn_lengths = batch['drawer_id_lengths']

        prev_image = torch.FloatTensor(batch['background'])
        prev_image = prev_image \
            .repeat(batch_size, 1, 1, 1)
        disc_prev_image = prev_image

        prev_image_real = torch.FloatTensor(batch['background_real'])
        prev_image_real = prev_image_real.repeat(batch_size, 1, 1, 1)
        

        # Initial inputs for the RNN set to zeros
        hidden = torch.zeros(1, batch_size, self.cfg.hidden_dim)
        prev_objects = torch.zeros(batch_size, self.cfg.num_objects)

        teller_images = []
        drawer_images = []
        added_entities = []

        drawer_dialog_losses = AverageMeter()
        #For Utterance_Generation
        enc_state = None
        for t in range(max_seq_len):
            #get current drawer utternace
            current_drawer_utterance = batch['drawer_turn_ids'][:, t, :]
            current_teller_drawer_utterance = batch['teller_drawer_turn_ids'][:, t, :]
            current_teller_drawer_utterance_len = batch['teller_drawer_id_lengths'][:, t]

            image = batch['image'][:, t]
            #added by Mingyang Zhou
            image_real = batch['image_real'][:, t]
            turns_word_embedding = batch['turn_word_embedding'][:, t]
            turns_lengths = batch['turn_lengths'][:, t]
            objects = batch['objects'][:, t]
            seq_ended = t > (batch['dialog_length'] - 1)
            
            ################################Utterance Generation################
            if t > 0:
                current_drawer_img_feat = self.drawer_img_encoder(batch['image'][:,t-1]) #Use the real image
            else:
                current_drawer_img_feat = self.drawer_img_encoder(prev_image)
            #encoder dialogs
            current_dialog_hidden, enc_state = self.dialog_encoder(current_teller_drawer_utterance, current_teller_drawer_utterance_len, initial_state=enc_state)
            preds, alphas = self.utterance_decoder(current_drawer_img_feat, current_drawer_utterance, enc_state=enc_state)
            targets = current_drawer_utterance[:, 1:].type(torch.LongTensor).cuda() 
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
            att_regularization = self.cfg.alpha_c * ((1 - alphas.sum(1))**2).mean()
            ################################Image Generation####################
            image_feature_map, image_vec, object_detections = \
                self.image_encoder(prev_image)
            _, current_image_feat, _ = self.image_encoder(image)

            turn_embedding = self.sentence_encoder(turns_word_embedding,
                                                   turns_lengths)
            rnn_condition, current_image_feat = \
                self.condition_encoder(turn_embedding,
                                       image_vec,
                                       current_image_feat)

            rnn_condition = rnn_condition.unsqueeze(0)
            # self.rnn.flatten_parameters()  # Added by Mingyang to Resolve the
            # Warning
            self.rnn.module.flatten_parameters()
            output, hidden = self.rnn(rnn_condition,
                                      hidden)

            output = output.squeeze(0)
            output = self.layer_norm(output)

            fake_image, mu, logvar, sigma = self._forward_generator(batch_size,
                                                                    output.detach(),
                                                                    image_feature_map)
            ######################################################################
            visualizer.track_sigma(sigma)

            hamming = objects - prev_objects
            hamming = torch.clamp(hamming, min=0)

            d_loss, d_real, d_fake, aux_loss, discriminator_gradient = \
                self._optimize_discriminator(image,
                                             fake_image.detach(),
                                             disc_prev_image,
                                             output,
                                             seq_ended,
                                             hamming,
                                             self.cfg.gp_reg,
                                             self.cfg.aux_reg)

            # append the segmentation loss accordingly
            if self.cfg.gan_type in ["recurrent_gan_drawer", "recurrent_gan_mingyang_img64_seg"]:
                #The size of seg_fake is adjusted to (Batch, N, C)
                seg_fake = fake_image.view(fake_image.size(0), fake_image.size(1), -1).permute(0,2,1)
                #The size of the seg_gt is obtained from image
                seg_gt = torch.argmax(image, dim=1).view(image.size(0), -1)

            else:
                assert self.cfg.seg_reg == 0, "the sge_reg must be equal to 0"
                seg_fake = None
                seg_gt = None

            #print(seg_gt)
            g_loss, generator_gradient = self._optimize_generator(fake_image, disc_prev_image.detach(), output.detach(), objects, self.cfg.aux_reg, seq_ended, mu, logvar, self.cfg.seg_reg, seg_fake, seg_gt)



            #compute drawer_dialog_loss + backwards##################
            drawer_dialog_loss = self.dialog_criterion(preds, targets)
            #print(drawer_dialog_loss.mean())
            drawer_dialog_loss += att_regularization
            drawer_dialog_loss = drawer_dialog_loss.mean()
            with torch.no_grad():
                total_caption_length = torch.sum(drawer_turn_lengths[:, t])
                #print(drawer_dialog_loss)
                drawer_dialog_losses.update(drawer_dialog_loss.item(),
                                     total_caption_length.item())
            drawer_dialog_loss.backward(retain_graph=True)
            self.dialog_optimizer.step()
            # reset dialog optimizer
            self.dialog_optimizer.zero_grad()
            
            ###########################################################
            if self.cfg.teacher_forcing:
                prev_image = image
            else:
                prev_image = fake_image

            disc_prev_image = image
            prev_objects = objects

            if (t + 1) % 2 == 0:
                prev_image = prev_image.detach()

            rnn_grads = []
            gru_grads = []
            condition_encoder_grads = []
            img_encoder_grads = []

            if t == max_seq_len - 1:
                rnn_gradient, gru_gradient, condition_gradient,\
                    img_encoder_gradient = self._optimize_rnn()

                rnn_grads.append(rnn_gradient.data.cpu().numpy())
                gru_grads.append(gru_gradient.data.cpu().numpy())
                condition_encoder_grads.append(
                    condition_gradient.data.cpu().numpy())

                if self.use_image_encoder:
                    img_encoder_grads.append(
                        img_encoder_gradient.data.cpu().numpy())

                visualizer.track(d_real, d_fake)

            hamming = hamming.data.cpu().numpy()[0]


            new_teller_images = []
            for x in image[:4].data.cpu():
                # print(x.shape)
                # new_x = self.unorm(x)
                # new_x = transforms.ToPILImage()(new_x).convert('RGB')
                # # new_x = np.array(new_x)[..., ::-1]
                # new_x = np.moveaxis(np.array(new_x), -1, 0)
                
                if self.cfg.image_gen_mode == "real":
                    new_x = self.unormalize(x)
                elif self.cfg.image_gen_mode == "segmentation":
                    new_x = self.unormalize_segmentation(x.data.numpy())
                elif self.cfg.image_gen_mode == "segmentation_onehot":
                    #TODO: Implement the functino to convert new_x to colored_image
                    new_x = self.unormalize_segmentation_onehot(x.data.cpu().numpy())
                    #print(new_x.shape)
                    #return

                # print(new_x.shape)
                new_teller_images.append(new_x)
            teller_images.extend(new_teller_images)

            new_drawer_images = []
            for x in fake_image[:4].data.cpu():
                # print(x.shape)
                # new_x = self.unorm(x)
                # new_x = transforms.ToPILImage()(new_x).convert('RGB')
                # # new_x = np.array(new_x)[..., ::-1]
                # new_x = np.moveaxis(np.array(new_x), -1, 0)
                
                if self.cfg.image_gen_mode == "real":
                    new_x = self.unormalize(x)
                elif self.cfg.image_gen_mode == "segmentation":
                    new_x = self.unormalize_segmentation(x.data.cpu().numpy())
                elif self.cfg.image_gen_mode == "segmentation_onehot":
                    #TODO: Implement the functino to convert new_x to colored_image
                    new_x = self.unormalize_segmentation_onehot(x.data.cpu().numpy())


                # print(new_x.shape)
                new_drawer_images.append(new_x)
            drawer_images.extend(new_drawer_images)

            # teller_images.extend(image[:4].data.numpy())
            # drawer_images.extend(fake_image[:4].data.cpu().numpy())
            # entities = str.join(',', list(batch['entities'][hamming > 0]))
            # added_entities.append(entities)
            del preds, alphas, att_regularization
            del drawer_dialog_loss

        #update the dialog decoded loss
        drawer_dialog_losses.compute_mean()

        if iteration % self.cfg.vis_rate == 0:
            visualizer.histogram()
            self._plot_losses(visualizer, g_loss, d_loss, aux_loss, iteration)
            #plot the dialog decoded loss, added by Mingyang
            self._plot_drawer_dialog_losses(visualizer, drawer_dialog_losses.avg, iteration)
            rnn_gradient = np.array(rnn_grads).mean()
            gru_gradient = np.array(gru_grads).mean()
            condition_gradient = np.array(condition_encoder_grads).mean()
            img_encoder_gradient = np.array(img_encoder_grads).mean()
            rnn_grads, gru_grads = [], []
            condition_encoder_grads, img_encoder_grads = [], []
            self._plot_gradients(visualizer, rnn_gradient, generator_gradient,
                                 discriminator_gradient, gru_gradient, condition_gradient,
                                 img_encoder_gradient, iteration)
            self._draw_images(visualizer, teller_images, drawer_images, nrow=4)
            #self.logger.write(epoch, "{}/{}".format(iteration,total_iters), d_real, d_fake, d_loss, g_loss)
            remaining_time = str(datetime.timedelta(
                seconds=current_batch_t * (total_iters - iteration)))
            self.logger.write(epoch, "{}/{}".format(iteration, total_iters),
                              d_real, d_fake, d_loss, g_loss, expected_finish_time=remaining_time)
            if isinstance(batch['turn'], list):
                batch['turn'] = np.array(batch['turn']).transpose()

            visualizer.write(batch['turn'][0])
            # visualizer.write(added_entities, var_name='entities')
            teller_images = []
            drawer_images = []

        if iteration % self.cfg.save_rate == 0:
            path = os.path.join(self.cfg.log_path,
                                self.cfg.exp_name)

            # self._save(fake_image[:4], path, epoch,
            #            iteration)
            if not self.cfg.debug:
                self.save_model(path, epoch, iteration)

    def _forward_generator(self, batch_size, condition, image_feature_maps):
        noise = torch.FloatTensor(batch_size,
                                  self.cfg.noise_dim).normal_(0, 1).cuda()

        fake_images, mu, logvar, sigma = self.generator(noise, condition,
                                                        image_feature_maps)

        return fake_images, mu, logvar, sigma

    def _optimize_discriminator(self, real_images, fake_images, prev_image,
                                condition, mask, objects, gp_reg=0, aux_reg=0):
        """Discriminator is updated every step independent of batch_size
        RNN and the generator
        """
        wrong_images = torch.cat((real_images[1:],
                                  real_images[0:1]), dim=0)
        wrong_prev = torch.cat((prev_image[1:],
                                prev_image[0:1]), dim=0)

        self.discriminator.zero_grad()
        real_images.requires_grad_()

        d_real, aux_real, _ = self.discriminator(real_images, condition,
                                                 prev_image)
        d_fake, aux_fake, _ = self.discriminator(fake_images, condition,
                                                 prev_image)
        d_wrong, _, _ = self.discriminator(wrong_images, condition,
                                           wrong_prev)

        d_loss, aux_loss = self._discriminator_masked_loss(d_real,
                                                           d_fake,
                                                           d_wrong,
                                                           aux_real,
                                                           aux_fake, objects,
                                                           aux_reg, mask)

        d_loss.backward(retain_graph=True)
        if gp_reg:
            reg = gp_reg * self._masked_gradient_penalty(d_real, real_images,
                                                         mask)
            reg.backward(retain_graph=True)

        grad_norm = _recurrent_gan.get_grad_norm(
            self.discriminator.parameters())
        self.discriminator_optimizer.step()

        d_loss_scalar = d_loss.item()
        d_real_np = d_real.cpu().data.numpy()
        d_fake_np = d_fake.cpu().data.numpy()
        aux_loss_scalar = aux_loss.item() if isinstance(
            aux_loss, torch.Tensor) else aux_loss
        grad_norm_scalar = grad_norm.item()
        del d_loss
        del d_real
        del d_fake
        del aux_loss
        del grad_norm
        gc.collect()

        return d_loss_scalar, d_real_np, d_fake_np, aux_loss_scalar, grad_norm_scalar

    def _optimize_generator(self, fake_images, prev_image, condition, objects, aux_reg,
                            mask, mu, logvar, seg_reg=0, seg_fake=None, seg_gt=None):
        self.generator.zero_grad()
        d_fake, aux_fake, _ = self.discriminator(fake_images, condition,
                                                 prev_image)
        g_loss = self._generator_masked_loss(d_fake, aux_fake, objects,
                                             aux_reg, mu, logvar, mask, seg_reg, seg_fake, seg_gt)

        g_loss.backward(retain_graph=True)
        gen_grad_norm = _recurrent_gan.get_grad_norm(
            self.generator.parameters())

        self.generator_optimizer.step()

        g_loss_scalar = g_loss.item()
        gen_grad_norm_scalar = gen_grad_norm.item()

        del g_loss
        del gen_grad_norm
        gc.collect()

        return g_loss_scalar, gen_grad_norm_scalar

    def _optimize_rnn(self):
        torch.nn.utils.clip_grad_norm_(
            self.rnn.parameters(), self.cfg.grad_clip)
        rnn_grad_norm = _recurrent_gan.get_grad_norm(self.rnn.parameters())
        self.rnn_optimizer.step()
        self.rnn.zero_grad()

        gru_grad_norm = None
        torch.nn.utils.clip_grad_norm_(
            self.sentence_encoder.parameters(), self.cfg.grad_clip)
        gru_grad_norm = _recurrent_gan.get_grad_norm(
            self.sentence_encoder.parameters())
        self.sentence_encoder_optimizer.step()
        self.sentence_encoder.zero_grad()

        ce_grad_norm = _recurrent_gan.get_grad_norm(
            self.condition_encoder.parameters())
        ie_grad_norm = _recurrent_gan.get_grad_norm(
            self.image_encoder.parameters())
        self.feature_encoders_optimizer.step()
        self.condition_encoder.zero_grad()
        self.image_encoder.zero_grad()
        return rnn_grad_norm, gru_grad_norm, ce_grad_norm, ie_grad_norm

    def _discriminator_masked_loss(self, d_real, d_fake, d_wrong, aux_real, aux_fake,
                                   objects, aux_reg, mask):
        """Accumulates losses only for sequences that have not ended
        to avoid back-propagation through padding"""
        d_loss = []
        aux_losses = []
        for b, ended in enumerate(mask):
            if not ended:
                sample_loss = self.criterion.discriminator(d_real[b], d_fake[b], d_wrong[b],
                                                           self.cfg.wrong_fake_ratio)
                if aux_reg > 0:
                    aux_loss = aux_reg * (self.aux_criterion(aux_real[b], objects[b]).mean() +
                                          self.aux_criterion(aux_fake[b], objects[b]).mean())
                    sample_loss += aux_loss
                    aux_losses.append(aux_loss)

                d_loss.append(sample_loss)

        d_loss = torch.stack(d_loss).mean()

        if len(aux_losses) > 0:
            aux_losses = torch.stack(aux_losses).mean()
        else:
            aux_losses = 0

        return d_loss, aux_losses

    def _generator_masked_loss(self, d_fake, aux_fake, objects, aux_reg,
                               mu, logvar, mask, seg_reg=0, seg_fake=None, seg_gt=None):
        """Accumulates losses only for sequences that have not ended
        to avoid back-propagation through padding
        Append the segmentation loss to the model.
        seg_fake: (1*C*H*W)
        seg_gt: (1*H*W)
        """
        g_loss = []
        for b, ended in enumerate(mask):
            if not ended:
                sample_loss = self.criterion.generator(d_fake[b])
                if aux_reg > 0:
                    aux_loss = aux_reg * \
                        self.aux_criterion(aux_fake[b], objects[b]).mean()
                else:
                    aux_loss = 0
                if mu is not None:
                    kl_loss = self.cfg.cond_kl_reg * \
                        kl_penalty(mu[b], logvar[b])
                else:
                    kl_loss = 0
                #Append a seg_loss to the total generator loss
                if seg_reg > 0:
                    #TODO: Implement the Segmentation Loss here
                    seg_loss = seg_reg * self.seg_criterion(seg_fake[b], seg_gt[b]) #By default it should just give a mean number
                    #print(seg_loss)
                else:
                    seg_loss = 0

                g_loss.append(sample_loss + aux_loss + kl_loss + seg_loss)

        g_loss = torch.stack(g_loss)
        return g_loss.mean()

    def _masked_gradient_penalty(self, d_real, real_images, mask):
        gp_reg = gradient_penalty(d_real, real_images).mean()
        return gp_reg

    # region Helpers
    def _plot_losses(self, visualizer, g_loss, d_loss, aux_loss,
                     iteration):
        _recurrent_gan._plot_losses(self, visualizer, g_loss, d_loss,
                                    aux_loss, iteration)

    def _plot_gradients(self, visualizer, rnn, gen, disc, gru, ce,
                        ie, iteration):
        _recurrent_gan._plot_gradients(self, visualizer, rnn, gen, disc,
                                       gru, ce, ie, iteration)

    def _draw_images(self, visualizer, real, fake, nrow):
        _recurrent_gan.draw_images(self, visualizer, real, fake, nrow)

    def _save(self, fake, path, epoch, iteration):
        _recurrent_gan._save(self, fake, path, epoch, iteration)

    def save_model(self, path, epoch, iteration):
        _recurrent_gan.save_drawer_model(self, path, epoch, iteration)

    def load_model(self, snapshot_path):
        _recurrent_gan.load_drawer_model(self, snapshot_path)
    # endregion
    def _plot_drawer_dialog_losses(self, visualizer, drawer_dialog_loss, iteration):
        visualizer.plot('Drawer Utt Decoder Loss', 'train', iteration, drawer_dialog_loss)
    def unormalize(self, x):
        """
        unormalize the image
        """
        new_x = self.unorm(x)
        new_x = transforms.ToPILImage()(new_x).convert('RGB')
        # new_x = np.array(new_x)[..., ::-1]
        new_x = np.moveaxis(np.array(new_x), -1, 0)
        return new_x

    def unormalize_segmentation(self, x):
        new_x = (x + 1) * 127.5
        # new_x = new_x.transpose(1, 2, 0)[..., ::-1]
        return new_x
    def unormalize_segmentation_onehot(self, x):
        """
        Convert the segmentation into image
        """

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
        seg_map = np.argmax(x, axis=0)
        new_x = np.zeros((3, seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
        for i in range(seg_map.shape[0]):
            for j in range(seg_map.shape[1]):
                new_x[:,i,j] = LABEL2COLOR[seg_map[i,j]]["color"]
        return new_x
