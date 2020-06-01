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

# Tester for GanDraw_Baseline1


class DrawerTester():

    def __init__(self, cfg, use_val=False, use_test=False, iteration=None, test_eval=False, visualize_batch=0, visualize_images=[]):
        self.model = INFERENCE_MODELS[cfg.gan_type](cfg)

        if use_val:
            dataset_path = cfg.val_dataset
            #dataset_path = cfg.test_dataset
            model_path = os.path.join(cfg.log_path, cfg.exp_name)
        elif use_test:
            dataset_path = cfg.test_dataset
            model_path = os.path.join(cfg.log_path, cfg.exp_name)
        else:
            dataset_path = cfg.dataset
            model_path = cfg.load_snapshot
        
        if test_eval:
            dataset_path = cfg.test_dataset
            model_path = cfg.load_snapshot

        self.model.load(model_path, iteration)
        self.dataset = DATASETS[cfg.dataset](path=keys[dataset_path],
                                             cfg=cfg,
                                             img_size=cfg.img_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=cfg.num_workers,
                                     drop_last=True)

        self.iterations = len(self.dataset) // 1
        # add current_iteration
        self.current_iteration = iteration

        if cfg.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data
        elif cfg.dataset in ['gandraw', 'gandraw_clean', 'gandraw_64', 'gandraw_64_DA']:
            self.dataloader.collate_fn = gandraw_dataset.collate_data

        if cfg.results_path is None:
            cfg.results_path = os.path.join(cfg.log_path, cfg.exp_name,'results')
        #print(cfg.results_path)
        if not os.path.exists(cfg.results_path):
            os.mkdir(cfg.results_path)

        self.cfg = cfg
        self.dataset_path = dataset_path

        self.visualize_batch = visualize_batch
        # Keep all the progress images to be processed.
        self.visualize_images = visualize_images

        self.word2index = self.dataset.word2index
        self.index2word = self.dataset.index2word
    def test(self, visualizer=None):
        i = 0
        references = []
        hypothesis = []
        for batch in tqdm(self.dataloader, total=self.iterations):
            if i == 0:
                visualize_progress = True
            else:
                visualize_progress = False
            self.model.predict(
                batch, batch_index = i, iteration=self.current_iteration, visualize_batch=0, visualize_progress=visualize_progress, visualize_images=self.visualize_images, visualizer=visualizer, word2index=self.word2index, index2word=self.index2word, references=references, hypothesis=hypothesis)
            i += 1

            #return
        bleu_1 = corpus_bleu(references, hypothesis, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypothesis,
                             weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypothesis,
                             weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypothesis)

        print('BLEU-1: {}, BLEU-2: {}, BLEU-3: {}, BLEU-4: {}'.format(bleu_1,
                                                                      bleu_2, bleu_3, bleu_4))
        
        # Plot the BLEU Score and Validation Loss
        if visualizer is not None:
            visualizer.plot("BLEU_4", "val", self.current_iteration, bleu_4 * 100)
            visualizer.plot("BLEU_3", "val", self.current_iteration, bleu_3 * 100)
            visualizer.plot("BLEU_2", "val", self.current_iteration, bleu_2 * 100)
            visualizer.plot("BLEU_1", "val", self.current_iteration, bleu_1 * 100)
