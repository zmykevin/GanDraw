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

class InferenceTeller():

    def __init__(self, cfg):
        super(InferenceTeller, self).__init__()

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
        #print(snapshot.keys())
        self.img_encoder.load_state_dict(
            snapshot['img_encoder_state_dict'])
        self.utterance_decoder.load_state_dict(
            snapshot['utterance_decoder_state_dict'])
        self.dialog_encoder.load_state_dict(
            snapshot['teller_dialog_encoder_state_dict']),
        self.optimizer.load_state_dict(snapshot['teller_optimizer_state_dict'])
