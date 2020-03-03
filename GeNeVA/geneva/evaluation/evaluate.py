# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Evaluation script."""
import torch
from random import randint

from geneva.utils.config import keys, parse_config
from geneva.evaluation.evaluate_metrics import report_inception_objects_score
from geneva.utils.visualize import VisdomPlotter
from geneva.inference.test import Tester
from geneva.inference.test_gandraw_baseline1 import GanDraw_Baseline1_Tester
from geneva.inference.test_gandraw_teller import TellerTester
from geneva.data.datasets import DATASETS


class Evaluator():

    @staticmethod
    def factory(cfg, visualizer, logger, visualize_images):
        # if cfg.gan_type == 'recurrent_gan':
        if cfg.gan_type in ['recurrent_gan']:  # Added by Mingyang
            return RecurrentGANEvaluator(cfg, visualizer, logger)
        # Added by Mingyang
        if cfg.gan_type in ['recurrent_gan_mingyang', 'recurrent_gan_mingyang_img64', 'recurrent_gan_stackGAN']:
            return GanDraw_Baseline1_Evaluator(cfg, visualizer, logger, visualize_images)
        if cfg.gan_type in ['recurrent_gan_teller']:
            return TellerEvaluator(cfg, visualizer, logger)


class RecurrentGANEvaluator():

    def __init__(self, cfg, visualizer, logger):
        self.cfg = cfg
        self.visualizer = visualizer
        self.logger = logger

    def evaluate(self, iteration):
        tester = Tester(self.cfg, use_val=True, iteration=iteration)
        tester.test()
        del tester
        torch.cuda.empty_cache()
        metrics_report = dict()
        if self.cfg.metric_inception_objects:
            io_jss, io_ap, io_ar, io_af1, io_cs, io_gs = \
                report_inception_objects_score(self.visualizer,
                                               self.logger,
                                               iteration,
                                               self.cfg.results_path,
                                               keys[self.cfg.dataset +
                                                    '_inception_objects'],
                                               keys[self.cfg.val_dataset],
                                               self.cfg.dataset)

            metrics_report['jaccard'] = io_jss
            metrics_report['precision'] = io_ap
            metrics_report['recall'] = io_ar
            metrics_report['f1'] = io_af1
            metrics_report['cossim'] = io_cs
            metrics_report['relsim'] = io_gs
        return metrics_report


class GanDraw_Baseline1_Evaluator():

    def __init__(self, cfg, visualizer, logger, visualize_images):
        self.cfg = cfg
        self.visualizer = visualizer
        self.logger = logger

        # Set a batch_index for progress image visualization
        self.val_dataset = DATASETS[cfg.dataset](path=keys[cfg.val_dataset],
                                                 cfg=cfg,
                                                 img_size=cfg.img_size)
        # print("length of the dataset: {}".format(len(self.val_dataset)))
        self.visualize_batch = randint(
            0, cfg.batch_size)
        self.visualize_images = visualize_images

    def evaluate(self, iteration):
        tester = GanDraw_Baseline1_Tester(
            self.cfg, use_val=True, iteration=iteration, visualize_batch=self.visualize_batch, visualize_images=self.visualize_images)
        tester.test(visualizer=self.visualizer)
        del tester
        torch.cuda.empty_cache()
        # TODO: compute the evaluation metrics
        # print("length of visualize images are: {}".format(
        #     len(self.visualize_images)))
        print("Compute the Evaluation Score on the Generate Images")
        metrics_report = {"summary": "eval report"}
        return metrics_report


class TellerEvaluator():

    def __init__(self, cfg, visualizer, logger):
        self.cfg = cfg
        self.visualizer = visualizer
        self.logger = logger

    def evaluate(self, iteration, model=None):
        tester = TellerTester(self.cfg, use_val=True,
                              iteration=iteration, model=model)
        tester.test(iteration=iteration, visualizer=self.visualizer)
        del tester
        print("Compute the Evaluation Score on the Teller")
        metrics_report = {"summary": "eval report"}
        return metrics_report

if __name__ == '__main__':
    cfg = parse_config()
    visualizer = VisdomPlotter(env_name=cfg.exp_name)
    logger = None
    dataset = cfg.dataset
    evaluator = Evaluator(cfg, visualizer, logger, dataset)
    evaluator.evaluate()