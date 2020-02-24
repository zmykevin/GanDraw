import datetime
from torch.nn import DataParallel

from geneva.models.teller_image_encoder import TellerImageEncoder
from geneva.models.teller_dialog_encoder import TellerDialogEncoder
from geneva.models.teller_dialog_decoder import TellerDialogDecoder

import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.nn.utils.rnn import pack_padded_sequence


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


class Teller():

    def __init__(self, cfg):
        super(Teller, self).__init__()

        ########################We will need Three Components########
        self.cfg = cfg
        # 1. Image Encoder
        self.img_encoder = DataParallel(TellerImageEncoder(cfg=cfg)).cuda()
        # self.img_encoder =
        # TellerImageEncoder(network=cfg.teller_img_encoder_net)

        # 2. Dialog Encoder
        self.dialog_encoder = DataParallel(TellerDialogEncoder(cfg=cfg)).cuda()

        # 2. Caption Decoder
        self.utterance_decoder = DataParallel(
            TellerDialogDecoder(cfg=cfg)).cuda()

        # 3. Changed_Objects_Predictor

        # 4. Teller's Optimizer
        parameters = []
        # Add all the  parameters  into  the optimizer
        if cfg.teller_img_encoder_net == "cnn":
            parameters.extend(self.img_encoder.parameters())

        else:
            self.img_encoder.eval()

        parameters.extend(self.utterance_decoder.parameters())
        parameters.extend(self.dialog_encoder.parameters())

        self.optimizer = optim.Adam(parameters, lr=cfg.teller_lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,  cfg.teller_step_size)
        # 5. Define the Loss Function
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    def train_batch(self, batch, epoch, iteration, visualizer, logger, total_iters=0, current_batch_t=0):

        batch_size = len(batch['image'])
        max_seq_len = batch['image'].size(1)
        teller_turn_lengths = batch['teller_id_lengths']

        # Make all the modules into training state
        if self.cfg.teller_img_encoder_net == "cnn":
            self.img_encoder.train()

        self.utterance_decoder.train()
        self.dialog_encoder.train()
        self.utterance_decoder.module.set_tf(True)

        teller_losses = AverageMeter()
        background_image = batch['background'].repeat(batch_size, 1, 1, 1)
        #print("background_image size is: {}".format(background_image.shape))
        for t in range(max_seq_len + 1):
            # zero the optimizer
            self.optimizer.zero_grad()
            current_target_img = batch['target_image'][:, 0]
            current_target_img_feat = self.img_encoder(current_target_img)
            current_teller_utterance = batch['teller_turn_ids'][:, t, :]
            # print(current_teller_utterance.size())

            # Encode the current drawer's image and gt image
            enc_state = None
            if t > 0:
                current_drawer_img = batch['image'][
                    :, t - 1]  # (batch_size, color_channel, )
                current_img_feat = self.img_encoder(current_drawer_img)
                # Input for dialog Encoder
                current_teller_drawer_utterance = batch[
                    'teller_drawer_turn_ids'][:, t - 1, :]
                current_teller_drawer_utterance_len = batch[
                    'teller_drawer_id_lengths'][:, t - 1]
                current_dialog_hidden, enc_state = self.dialog_encoder(
                    current_teller_drawer_utterance, current_teller_drawer_utterance_len, initial_state=enc_state)
            else:
                # If t is equal to 0
                # current_img_feat = torch.zeros(
                #     current_target_img_feat.size(), dtype=torch.float).cuda()
                current_img_feat = self.img_encoder(
                    background_image)

            # Fuse the two features through concatenation
            if self.cfg.teller_fuse == "concat":
                concatenated_feature = torch.cat([current_target_img_feat, current_img_feat], dim=2, out=None)
            elif self.cfg.teller_fuse == "elemwise_add":
                concatenated_feature = current_target_img_feat + current_img_feat

            # Run Decoder
            preds, alphas = self.utterance_decoder(
                concatenated_feature, current_teller_utterance, enc_state=enc_state)
            targets = current_teller_utterance[:, 1:].type(
                torch.LongTensor).cuda()  # convert the targets to GPU tensors

            targets = pack_padded_sequence(
                targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(
                preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
            att_regularization = self.cfg.alpha_c * \
                ((1 - alphas.sum(1))**2).mean()

            # print(targets.type())
            # print(preds.type())
            # compute the loss
            teller_loss = self.cross_entropy_loss(preds, targets)
            teller_loss += att_regularization

            with torch.no_grad():
                total_caption_length = torch.sum(teller_turn_lengths[:, t])
                teller_losses.update(teller_loss.item(),
                                     total_caption_length.item())

            # backpropagate the loss

            teller_loss.backward()
            self.optimizer.step()
            del preds
            del alphas
            del att_regularization
            del current_target_img_feat, current_img_feat, concatenated_feature
            del teller_loss

        # Average the loss
        # teller_loss = teller_loss / (max_seq_len+1)
        teller_losses.compute_mean()
        # Log and Visualization
        if iteration % self.cfg.vis_rate == 0:
            # Plot the loss in the visdom plot
            self._plot_teller_losses(visualizer, teller_losses.avg, iteration)
            remaining_time = str(datetime.timedelta(
                seconds=current_batch_t * (total_iters - iteration)))
            # Log the loss information
            print("Epoch: {epoch}, {iteration}/{total_iters}, teller_loss: {teller_loss}, expected_finish_time: {expected_finish_time}".format(epoch=epoch,

                                                                                                                                               iteration=iteration,
                                                                                                                                               total_iters=total_iters,
                                                                                                                                               teller_loss=teller_losses.avg, expected_finish_time=remaining_time))
        # Save the model
        if iteration % self.cfg.save_rate == 0:
            path = os.path.join(self.cfg.log_path,
                                self.cfg.exp_name)
            if not self.cfg.debug:
                self.save_model(path, epoch, iteration)

        del teller_losses

    def _plot_teller_losses(self, visualizer, teller_loss, iteration):
        visualizer.plot('Teller Decoder Loss', 'train', iteration, teller_loss)

    def save_model(self, path, epoch, iteration):
        if not os.path.exists(path):
            os.mkdir(path)
        snapshot = {
            'epoch': epoch,
            'iteration': iteration,
            'img_encoder_state_dict': self.img_encoder.state_dict(),
            'teller_dialog_encoder_state_dict': self.dialog_encoder.state_dict(),
            'utterance_decoder_state_dict': self.utterance_decoder.state_dict(),
            'teller_optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg,
        }

        torch.save(snapshot, '{}/snapshot_{}.pth'.format(path, iteration))

    def load_model(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.img_encoder.load_state_dict(
            snapshot['img_encoder_state_dict'])
        self.utterance_decoder.load_state_dict(
            snapshot['utterance_decoder_state_dict'])
        self.dialog_encoder.load_state_dict(
            snapshot['teller_dialog_encoder_state_dict']),
        self.optimizer.load_state_dict(snapshot['optimizer'])
