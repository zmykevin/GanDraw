import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from geneva.data.datasets import DATASETS
from geneva.evaluation.evaluate_metrics import report_inception_objects_score
from geneva.utils.config import keys, parse_config
from geneva.models.models import INFERENCE_MODELS
from geneva.data import codraw_dataset
from geneva.data import clevr_dataset
from geneva.data import gandraw_dataset

from nltk.translate.bleu_score import corpus_bleu
import random
import torch.nn as nn
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


class TellerTester():

    def __init__(self, cfg, use_val=False, iteration=None, test_eval=False, model=None):
        self.model = model
        # eval
        self.model.img_encoder.eval()
        self.model.utterance_decoder.eval()
        self.model.dialog_encoder.eval()
        self.model.utterance_decoder.module.set_tf(False)

        if use_val:
            dataset_path = cfg.val_dataset
            model_path = os.path.join(cfg.log_path, cfg.exp_name)
        else:
            dataset_path = cfg.dataset
            model_path = cfg.load_snapshot
        if test_eval:
            dataset_path = cfg.test_dataset
            model_path = cfg.load_snapshot

        # self.model.load(model_path, iteration), not loadning the model
        self.dataset = DATASETS[cfg.dataset](path=keys[dataset_path],
                                             cfg=cfg,
                                             img_size=cfg.img_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=cfg.num_workers,
                                     drop_last=True)

        self.iterations = len(self.dataset) // cfg.batch_size

        if cfg.dataset in ['codraw', 'codrawDialog']:
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == "gandraw":
            self.dataloader.collate_fn = gandraw_dataset.collate_data

        if cfg.results_path is None:
            cfg.results_path = os.path.join(cfg.log_path, cfg.exp_name,
                                            'results')
            if not os.path.exists(cfg.results_path):
                os.mkdir(cfg.results_path)

        self.cfg = cfg
        self.dataset_path = dataset_path

        # load the word2index and index2word
        self.word2index = self.dataset.word2index
        self.index2word = self.dataset.index2word

        # 5. Define the Loss Function
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    def test(self, iteration=0, visualizer=None):
        #set teacher forcing to "False"

        with torch.no_grad():
            # Randomly Sample an Index to do Qualitative Example
            #sampled_index = random.randint(0,len(self.dataloader))
            sampled_index = 0
            references = []
            hypothesis = []
            sampled_conversation = []
            for i, batch in enumerate(self.dataloader):
                batch_size = len(batch['image'])
                max_seq_len = batch['image'].size(1)
                teller_turn_lengths = batch['teller_id_lengths']

                teller_val_losses = AverageMeter()
                for t in range(max_seq_len + 1):
                    current_target_img = batch['target_image'][:, 0]
                    current_target_img_feat = self.model.img_encoder(
                        current_target_img)
                    current_teller_utterance = batch[
                        'teller_turn_ids'][:, t, :]
                    # print(current_teller_utterance.size())

                    enc_state = None
                    if t > 0:
                        current_drawer_img = batch['image'][
                            :, t - 1]  # (batch_size, color_channel, )
                        current_img_feat = self.model.img_encoder(
                            current_drawer_img)
                        # Input for dialog Encoder
                        current_teller_drawer_utterance = batch[
                            'teller_drawer_turn_ids'][:, t - 1, :]
                        current_teller_drawer_utterance_len = batch[
                            'teller_drawer_id_lengths'][:, t - 1]
                        #current_dialog_hidden, enc_state = self.model.dialog_encoder(current_teller_drawer_utterance, current_teller_drawer_utterance_len, initial_state = enc_state)
                        current_dialog_hidden, enc_state = self.model.dialog_encoder(
                            current_teller_drawer_utterance, current_teller_drawer_utterance_len)
                    else:
                        # If t is equal to 0
                        current_img_feat = torch.zeros(
                            current_target_img_feat.size(), dtype=torch.float).cuda()

                    # Fuse the two features through concatenation

                    # concatenated_feature = torch.cat(
                    #     [current_target_img_feat, current_img_feat], dim=2, out=None)
                    if self.cfg.teller_fuse == "concat":
                        concatenated_feature = torch.cat([current_target_img_feat, current_img_feat], dim=2, out=None)
                    elif self.cfg.teller_fuse == "elemwise_add":
                        concatenated_feature = current_target_img_feat + current_img_feat

                    # Compute BLEU Score and other eval metrics
                    preds, alphas = self.model.utterance_decoder(
                        concatenated_feature, current_teller_utterance)
                    targets = current_teller_utterance[
                        :, 1:].type(torch.LongTensor).cuda()

                    targets = pack_padded_sequence(
                        targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
                    packed_preds = pack_padded_sequence(
                        preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
                    att_regularization = self.cfg.alpha_c * \
                        ((1 - alphas.sum(1))**2).mean()

                    teller_val_loss = self.cross_entropy_loss(
                        packed_preds, targets)
                    teller_val_loss += att_regularization

                    total_caption_length = torch.sum(
                        teller_turn_lengths[:, t]).item()
                    teller_val_losses.update(
                        teller_val_loss, total_caption_length)

                    for utterance_set in current_teller_utterance.cpu().numpy().tolist():
                        refs = []
                        refs.append([idx for idx in utterance_set
                                     if idx != self.word2index['<s_start>'] and idx != self.word2index['<pad>']])
                        references.append(refs)

                    # compuate the BLEU score
                    word_idxs = torch.max(preds, dim=2)[1].cpu().numpy()
                    for idxs in word_idxs.tolist():
                        hypothesis.append([idx for idx in idxs
                                           if idx != self.word2index['<s_start>'] and idx != self.word2index['<pad>']])

                    # Genearate the sample caption
                    if i == sampled_index:
                        concatenated_feature_beam = concatenated_feature.expand(
                            self.cfg.teller_beamsize, concatenated_feature.size(1), concatenated_feature.size(2))
                        if t > 0:
                            enc_state_beam = (enc_state[0].expand(self.cfg.teller_beamsize, enc_state[0].size(1), enc_state[0].size(2)),
                                              enc_state[1].expand(self.cfg.teller_beamsize, enc_state[
                                                                  1].size(1), enc_state[1].size(2))
                                              )
                        else:
                            enc_state_beam = None
                        utterance_decoder = self.model.utterance_decoder.module
                        sentence, alpha = utterance_decoder.caption(
                            concatenated_feature_beam, self.cfg.teller_beamsize, enc_state=enc_state_beam)
                        #sentence, alpha = self.model.utterance_decoder.module.caption(concatenated_feature_beam, self.cfg.teller_beamsize)
                        sentence_tokens = []
                        for word_idx in sentence:
                            if word_idx != self.word2index['<s_start>'] and word_idx != self.word2index['<pad>'] and word_idx != self.word2index['<s_end>']:
                                sentence_tokens.append(
                                    self.index2word[word_idx])
                            if word_idx == self.word2index['<s_end>']:
                                break
                        sampled_conversation.append(sentence_tokens)
                        # print(sentence_tokens)

                    # Visualize the sampled conversation in Visualizer
                    if visualizer is not None:
                        # for sentence_tokens in sampled_conversation:
                        #     visualizer.text(
                        #         'Teller: ' + ' '.join(sentence_tokens))
                        visualizer.write_dialog(sampled_conversation)

            teller_val_losses.compute_mean()
            bleu_1 = corpus_bleu(references, hypothesis, weights=(1, 0, 0, 0))
            bleu_2 = corpus_bleu(references, hypothesis,
                                 weights=(0.5, 0.5, 0, 0))
            bleu_3 = corpus_bleu(references, hypothesis,
                                 weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = corpus_bleu(references, hypothesis)

            print('BLEU-1: {}, BLEU-2: {}, BLEU-3: {}, BLEU-4: {}'.format(bleu_1,
                                                                          bleu_2, bleu_3, bleu_4))
            print("val_loss is: {}".format(teller_val_losses.avg))
            
            # Plot the BLEU Score and Validation Loss
            if visualizer is not None:
                visualizer.plot("BLEU_4", "val", iteration, bleu_4 * 100)
                visualizer.plot("BLEU_3", "val", iteration, bleu_3 * 100)
                visualizer.plot("BLEU_2", "val", iteration, bleu_2 * 100)
                visualizer.plot("BLEU_1", "val", iteration, bleu_1 * 100)
                print(type(teller_val_losses.avg.item()))
                visualizer.plot("Teller Decoder Val Loss", 'val', iteration, teller_val_losses.avg.item())
