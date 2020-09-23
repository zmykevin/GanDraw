import pprint
import json
import easydict 

from geneva.data.datasets import DATASETS
from geneva.evaluation.evaluate import Evaluator
from geneva.utils.config import keys, parse_config
from geneva.utils.visualize import VisdomPlotter
from geneva.models.models import MODELS
from geneva.data import gandraw_dataset

import time
if __name__ == '__main__':
	config_file = "example_args/gandraw_args.json"
	
	with open(config_file, 'r') as f:
		cfg = json.load(f)

	cfg = easydict.EasyDict(cfg)
	best_iteration = 1500 #Manually define

	#Initialize the evaluator
	visualizer = VisdomPlotter(env_name=cfg.exp_name, server=cfg.vis_server)
	logger = None

	evaluator = Evaluator.factory(cfg, visualizer, logger, visualize_images=[])
	metric_report = evaluator.evaluate(best_iteration, use_test=True)

	print("evaluation results for iter: {} on test data: \n".format(best_iteration))
	for key, value in metric_report.items():
		print("{metric_name}: {metric_value}; \n".format(metric_name=key, metric_value=value))
