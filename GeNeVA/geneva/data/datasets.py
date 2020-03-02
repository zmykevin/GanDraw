# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Datasets Manager"""
from geneva.data import clevr_dataset
from geneva.data import codraw_dataset
from geneva.data import gandraw_dataset

DATASETS = {
    'codraw': codraw_dataset.CoDrawDataset,
    'iclevr': clevr_dataset.ICLEVERDataset,
    'codrawDialog': codraw_dataset.CoDrawDialogDataset,
    'gandraw': gandraw_dataset.GanDrawDataset,
    'gandraw_clean': gandraw_dataset.GanDrawDataset,
    'gandraw_64': gandraw_dataset.GanDrawDataset,
    'gandraw_64_DA': gandraw_dataset.GanDrawDataset
}
