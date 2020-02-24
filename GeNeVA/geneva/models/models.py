# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Models Manager"""
from geneva.models.recurrent_gan import RecurrentGAN
from geneva.models.inference_models.recurrent_gan import InferenceRecurrentGAN

# Added by Mingyang
from geneva.models.recurrent_gan_mingyang import RecurrentGAN_Mingyang
from geneva.models.inference_models.recurrent_gan_mingyang import InferenceRecurrentGAN_Mingyang

# Model for Teller
from geneva.models.teller_mingyang import Teller
from geneva.models.drawer_mingyang import Drawer

MODELS = {
    'recurrent_gan': RecurrentGAN,
    'recurrent_gan_mingyang': RecurrentGAN_Mingyang,
    'recurrent_gan_mingyang_img64': RecurrentGAN_Mingyang,
    'recurrent_gan_stackGAN': RecurrentGAN_Mingyang,
    'recurrent_gan_teller': Teller,
    'recurrent_gan_drawer': Drawer}


INFERENCE_MODELS = {
    'recurrent_gan': InferenceRecurrentGAN,
    'recurrent_gan_mingyang': InferenceRecurrentGAN_Mingyang,
    'recurrent_gan_mingyang_img64': InferenceRecurrentGAN_Mingyang,
    'recurrent_gan_stackGAN': InferenceRecurrentGAN_Mingyang,
}
