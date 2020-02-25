# GanDraw
The Repository for the baselines for GanDraw Dataset
## Visdom Port
We Visualize the Training Progress with Visdom. The visdom port for each server is listed as following:
1. __nlp.cs.ucdavis.edu__: 8097
2. __interaction.cs.ucdavis.edu__: 8098

## GanDraw Baseline1 Exp Setting Instructions
### ImgDimension 128 vs 64
To switch between using a dataset with image dimension of 128 x 128 and a dataset with image dimensino of 64 x 64, you need to change the following in __gandraw_args.json__:
1. __img_size__: 128 -> 64
2. __dataset__: gandraw -> gandraw_64
3. __test_dataset__: gandraw_test -> gandraw_64_test
4. __val_dataset__: gandraw_val -> gandraw_64_val
5. __gan_type__: recurrent_gan_mingyang -> recurrent_gan_mingyang_img64

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
