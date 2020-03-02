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

# Tester for GanDraw_Baseline1


class GanDraw_Baseline1_Tester():

    def __init__(self, cfg, use_val=False, iteration=None, test_eval=False, visualize_batch=0, visualize_images=[]):
        self.model = INFERENCE_MODELS[cfg.gan_type](cfg)

        if use_val:
            dataset_path = cfg.val_dataset
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
                                     batch_size=cfg.batch_size,
                                     shuffle=False,
                                     num_workers=cfg.num_workers,
                                     drop_last=True)

        self.iterations = len(self.dataset) // cfg.batch_size
        # add current_iteration
        self.current_iteration = iteration

        if cfg.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data
        elif cfg.dataset in ['gandraw', 'gandraw_clean', 'gandraw_64', 'gandraw_64_DA']:
            self.dataloader.collate_fn = gandraw_dataset.collate_data

        if cfg.results_path is None:
            cfg.results_path = os.path.join(cfg.log_path, cfg.exp_name,
                                            'results')
            if not os.path.exists(cfg.results_path):
                os.mkdir(cfg.results_path)

        self.cfg = cfg
        self.dataset_path = dataset_path

        self.visualize_batch = visualize_batch
        # Keep all the progress images to be processed.
        self.visualize_images = visualize_images

    def test(self, visualizer=None):
        i = 0
        for batch in tqdm(self.dataloader, total=self.iterations):
            if i == 0:
                visualize_progress = True
            else:
                visualize_progress = False
            self.model.predict(
                batch, iteration=self.current_iteration, visualize_batch=self.visualize_batch, visualize_progress=visualize_progress, visualize_images=self.visualize_images, visualizer=visualizer)
            i += 1
