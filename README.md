# GanDraw
The Repository for the baselines for GanDraw Dataset

## GanDraw Baseline1 Exp Setting Instructions
### ImgDimension 128 vs 64
To switch between using a dataset with image dimension of 128 x 128 and a dataset with image dimensino of 64 x 64, you need to change the following in __gandraw_args.json__:
1. img_size: 128 -> 64
2. dataset: gandraw -> gandraw_64
3. test_dataset: gandraw_test -> gandraw_64_test
4. val_dataset: gandraw_val -> gandraw_64_val
5. gan_type: recurrent_gan_mingyang -> recurrent_gan_mingyang_img64

The __expriment__ and __results_path__ should also be changed accordingly.

## TODO:
1. Move the GeNeVa Baselines to This Repository. 
2. Create the Readme on preprocessing GANDRAW dataset. 
3. Creat the Readme on running Baseline1
4. Creat the Readme on running Teller
5. Create the Baselines for Drawer
6. Create the Readme on Drawer
7. Create the  Code for Baseline1 Evaluation
8. Create the Code for Teller Evaluation. 
9. Create the Code for Drawer Evaluation. 
