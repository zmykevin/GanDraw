"""
Convert the image from the generated segmentation map to GauGAN
"""
import numpy as np
import cv2
import os
from PIL import Image

L2NORML_RAW = {
            156: {"name": "sky", "index": 0},
            110: {"name": "dirt", "index": 1},
            124: {"name": "gravel", "index": 2},
            135: {"name": "mud", "index": 3},
            14: {"name": "sand", "index": 4},
            105: {"name": "clouds", "index": 5},
            119: {"name": "fog", "index": 6},
            126: {"name": "hill", "index": 7},
            134: {"name": "mountain", "index": 8},
            147: {"name": "river", "index": 9},
            149: {"name": "rock", "index": 10},
            154: {"name": "sea", "index": 11},
            158: {"name": "snow", "index": 12},
            161: {"name": "stone", "index": 13},
            177: {"name": "water", "index": 14},
            96: {"name": "bush", "index": 15},
            118: {"name": "flower", "index": 16},
            123: {"name": "grass", "index": 17},
            162: {"name": "straw", "index": 18},
            168: {"name": "tree", "index": 19},
            181: {"name": "wood", "index": 20},
            148: {"name": "road", "index": 21}
        }

NORML2COLOR_RAW = {
            0: {"name": "sky", "color": np.array([134, 193, 46])},
            1: {"name": "dirt", "color": np.array([30, 22, 100])},
            2: {"name": "gravel", "color": np.array([163, 164, 153])},
            3: {"name": "mud", "color": np.array([35, 90, 74])},
            4: {"name": "sand", "color": np.array([196, 15, 241])},
            5: {"name": "clouds", "color": np.array([198, 182, 115])},
            6: {"name": "fog", "color": np.array([76, 60, 231])},
            7: {"name": "hill", "color": np.array([190, 128, 82])},
            8: {"name": "mountain", "color": np.array([122, 101, 17])},
            9: {"name": "river", "color": np.array([97, 140, 33])},
            10: {"name": "rock", "color": np.array([90, 90, 81])},
            11: {"name": "sea", "color": np.array([255, 252, 51])},
            12: {"name": "snow", "color": np.array([51, 255, 252])},
            13: {"name": "stone", "color": np.array([106, 107, 97])},
            14: {"name": "water", "color": np.array([0, 255, 0])},
            15: {"name": "bush", "color": np.array([204, 113, 46])},
            16: {"name": "flower", "color": np.array([0, 0, 255])},
            17: {"name": "grass", "color": np.array([255, 0, 0])},
            18: {"name": "straw", "color": np.array([255, 51, 252])},
            19: {"name": "tree", "color": np.array([255, 51, 175])},
            20: {"name": "wood", "color": np.array([66, 18, 120])},
            21: {"name": "road", "color": np.array([255, 255, 0])},
        }

NORML2COLOR = {k: x["color"] for k, x in NORML2COLOR_RAW.items()}
NORML2LABEL = {x["index"]:k for k,x in L2NORML_RAW.items()}


def color2label(color):
    for k, x in NORML2COLOR.items():
        if x[0] == color[0] and x[1] == color[1] and x[2] == color[2]:
            return NORML2LABEL[k]
    return None
def segconverter(seg_im):
    """
    Convert the segmentation image to a 2D matrix with assigned label.
    Input is a numpy matrix
    """
    gen_result = np.zeros((seg_im.shape[0], seg_im.shape[1]), dtype=np.int32)
    #convert gen_raw to 2D matrix
    unique_colors = np.unique(seg_im.reshape(-1, seg_im.shape[2]), axis=0)
    #print(unique_colors)
    for c in unique_colors:
        c_index = np.where(np.all(seg_im == c, axis=-1))
        #
        c_label = color2label(c)
        assert c_label is not None, "label cannot be None"
        gen_result[c_index] = c_label
    return gen_result

if __name__ == "__main__":
	data_path = "/Users/kevinz/Desktop/UC_DAVIS/Research/Iterative-Image-Editing-Dialogue/sample_output/GanDraw/baseline1_segonehot_segaux/result"
	all_dir = os.listdir(data_path)
	all_dir.remove(".DS_Store")
	#data_gt_path = sorted([x for x in all_dir if re.search("gt", x)], key = lambda x: int(x.split('_')[0]))
	data_gen_path = sorted([x for x in all_dir if x not in data_gt_path], key = lambda x: int(x.split('_')[0]))
	#data_gt_path = [os.path.join(data_path,x) for x in data_gt_path]
	data_gen_path = [os.path.join(data_path,x) for x in data_gen_path]

	#extract corresponding_generation_file
	folder_index = []
	data_gen_files = []
	for gen_path in data_gen_path:
	    all_images = sorted([x for x in os.listdir(gen_path) if re.search(".png",x) and re.search("4500", x)], key = lambda x: int(x.split('_')[0]))        
	    data_gen_files.append([os.path.join(gen_path, x) for x in all_images])
	    folder_index.append(gen_path.split('/')[-1])

	for gen_im_paths, folder_i  in zip(data_gen_file, folder_i):
		#load the image
        for gen_im_path in gen_im_paths:
	        gen_raw = cv2.imread(gen_im_path)
	        gen_seg = cv2.cvtColor(gen_raw, cv2.COLOR_BGR2RGB)
	        gen_seg = segconverter(gen_raw)

	        #save gen_seg to the output_file
	        gen_seg_gaugan_path = gen_im_path.split('/')[-1]
	        gen_seg_gaugan_path = gen_seg_gaugan_path.split('.')[0]+"_gaugan.png"

	        #save the real image
            gen_seg_gaugan = Image.fromarray(np.uint8(gen_seg))
            labelimg.save(gen_seg_gaugan_path)


