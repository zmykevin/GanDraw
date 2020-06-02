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
        self.img_encoder.eval()
        # 2. Dialog Encoder
        self.dialog_encoder = DataParallel(TellerDialogEncoder(cfg=cfg)).cuda()
        self.dialog_encoder.eval()
        # 2. Caption Decoder
        self.utterance_decoder = DataParallel(
            TellerDialogDecoder(cfg=cfg)).cuda()
        self.utterance_decoder.eval()
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
        
        #For the interactive teller
        self.tgt_img = None
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
    def generate_utt(self, input_img, input_utt, input_utt_len, word2index, index2word):
        """
        Input_Img and Input_utt are already processed to the format accepted by the Model
        """
        with torch.no_grad():
            assert self.tgt_img is not None
            current_target_img_feat = self.img_encoder(self.tgt_img)
            if input_utt is not None:
                current_dialog_hidden, enc_state = self.dialog_encoder(input_utt, input_utt_len, self.prev_enc_state)
                #reset prev_enc_state
                self.prev_enc_state = enc_state
            
            if input_img is not None:
                current_img_feat = self.img_encoder(input_img)
            else:
                current_img_feat = torch.zeros(current_target_img_feat.size(), dtype=torch.float).cuda()
            #fuse the img feats
            if self.cfg.teller_fuse == "concat":
                concatenated_feature = torch.cat([current_target_img_feat, current_img_feat], dim=2, out=None)
            elif self.cfg.teller_fuse == "elemwise_add":
                concatenated_feature = current_target_img_feat + current_img_feat
            
            #generate the sample caption
            concatenated_feature_beam = concatenated_feature.expand(self.cfg.teller_beamsize, concatenated_feature.size(1), concatenated_feature.size(2))

            if input_img is not None:
                enc_state_beam = (enc_state[0].expand(self.cfg.teller_beamsize, enc_state[0].size(1), enc_state[0].size(2)),enc_state[1].expand(self.cfg.teller_beamsize, enc_state[1].size(1), enc_state[1].size(2)))
            else:
                enc_state_beam = None
            utterance_decoder = self.utterance_decoder.module
            sentence, alpha = utterance_decoder.caption(concatenated_feature_beam, self.cfg.teller_beamsize, enc_state=enc_state_beam)
            
            stop = False
            sentence_tokens = []
            for word_idx in sentence:
                if word_idx != word2index['<s_start>'] and word_idx != word2index['<pad>'] and word_idx != word2index['<s_end>']:
                    sentence_tokens.append(
                        index2word[word_idx])
                if word_idx == word2index['<s_end>']:
                    break
                if word_idx == word2index['<d_end>']:
                    stop = True
                    break
            output_utt = " ".join(sentence_tokens)
            return output_utt, stop

    def import_tgt_img(self, tgt_img):
        self.tgt_img = tgt_img
    def reset_teller(self):
        self.prev_enc_state = None
        self.tgt_img = None
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
