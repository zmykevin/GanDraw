import os
import glob

import torch
from torch.utils.data import DataLoader
import json  # Added to initialize the setting in Jupyter Notebook-by Mingyang
import easydict  # Added to initialize the setting in Jupyter Notebook-by Mingyang

from geneva.data.datasets import DATASETS
from geneva.evaluation.evaluate import Evaluator
from geneva.utils.config import keys, parse_config
from geneva.utils.visualize import VisdomPlotter
from geneva.models.models import MODELS
from geneva.data import codraw_dataset
from geneva.data import clevr_dataset
from geneva.data import gandraw_dataset
# from torch.nn import DataParallel #Added by Mingyang Zhou

import time


class Trainer():

    def __init__(self, cfg):
        img_path = os.path.join(cfg.log_path,
                                cfg.exp_name,
                                'train_images_*')
        if glob.glob(img_path):
            raise Exception('all directories with name train_images_* under '
                            'the experiment directory need to be removed')
        path = os.path.join(cfg.log_path, cfg.exp_name)

        self.model = MODELS[cfg.gan_type](cfg)
        # Added by Mingyang
#         if cfg.gan_type == "recurrent_gan_mingyang":
#           print("Wrap DataParallel Around the whole model")
#           self.model = DataParallel(self.model)

        self.model.save_model(path, 0, 0)

        if cfg.load_snapshot is not None:
            self.model.load_model(cfg.load_snapshot)
        shuffle = False

        print(keys[cfg.dataset])
        self.dataset = DATASETS[cfg.dataset](
            path=keys[cfg.dataset], cfg=cfg, img_size=cfg.img_size)

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=cfg.batch_size,
                                     shuffle=shuffle,
                                     num_workers=cfg.num_workers,
                                     pin_memory=True,
                                     drop_last=True)

        if cfg.dataset in ['codraw', 'codrawDialog']:
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data
        elif cfg.dataset in ["gandraw", "gandraw_clean", "gandraw_64", "gandraw_64_DA"]:
            self.dataloader.collate_fn = gandraw_dataset.collate_data

        self.visualizer = VisdomPlotter(
            env_name=cfg.exp_name, server=cfg.vis_server)
        self.logger = None

        self.cfg = cfg

    def train(self):
        iteration_counter = 0  # Last Iteration
        #print("Total number of training data: {}".format(len(self.dataset)))
        num_batches = len(self.dataloader)
        total_iterations = num_batches * self.cfg.epochs
        current_batch_time = 0  # Record the time it takes to process one batch
        #print("total iteration is: {}".format(total_iterations))
        visualize_images = []
        for epoch in range(self.cfg.epochs):
            if cfg.dataset in ['codraw', 'gandraw', "gandraw_64", "gandraw_64_DA"]:
                self.dataset.shuffle()

            for batch in self.dataloader:
                if cfg.gan_type == "recurrent_gan":
                    self.model.train_batch(batch,
                                           epoch,
                                           iteration_counter,
                                           self.visualizer,
                                           self.logger)
                else:
                    current_batch_start = time.time()
                    self.model.train_batch(batch,
                                           epoch,
                                           iteration_counter,
                                           self.visualizer,
                                           self.logger,
                                           total_iters=total_iterations,
                                           current_batch_t=current_batch_time
                                           )
                    current_batch_time = time.time() - current_batch_start
                    #print("batch_time is: {}".format(current_batch_time))

                if iteration_counter >= 0 and iteration_counter % self.cfg.save_rate == 0:
                    torch.cuda.empty_cache()
                    evaluator = Evaluator.factory(self.cfg, self.visualizer,
                                                  self.logger, visualize_images=visualize_images)
                    evaluator.evaluate(iteration_counter)
                    del evaluator

                iteration_counter += 1
                # if iteration_counter > 1:
                #return

if __name__ == '__main__':
    config_file = "example_args/gandraw_args.json"
    # Load the config_file
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    # convert cfg as easydict
    cfg = easydict.EasyDict(cfg)
    cfg.load_snapshot = None
    trainer = Trainer(cfg)
    print("Finishing Initilizing the Trainer")
    trainer.train()
