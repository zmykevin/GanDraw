from Interactive_Bot.TellerBot import GanDraw_Teller
from Interactive_Bot.DrawerBot import GanDraw_Silent_Drawer
from Interactive_Bot.DrawerTalkBot import GanDraw_Talkative_Drawer
from Interactive_Bot.interactive_utils import img_to_bytes, segmap_to_real
import easydict
import json
import cv2
import os
import random
from PIL import Image
import numpy as np
from geneva.evaluation.seg_scene_similarity_score import gaugancodraw_eval_metrics, mean_IoU

def load_cfg(config_file):
	with open(config_file, 'r') as f:
	    cfg = json.load(f)
	# convert cfg as easydict
	cfg = easydict.EasyDict(cfg)
	cfg.load_snapshot = None
	return cfg


def teller_drawer_silent_game(sample_dialog, output_path, teller, drawer, meanIoU_list, seg_similarity_list):
    sample_dialog_id = sample_dialog['target_image'].split('/')[-1][:-4]
    #mkdir
    current_output_path = '/'.join([output_path, sample_dialog_id])
    os.makedirs(current_output_path, exist_ok=True)
    current_dialog = {sample_dialog_id: {"dialog":[], "meanIoU": None, "seg_similarity": None}}

    tgt_img_path = '/'.join([data_path, sample_dialog["target_image"]])
    tgt_img_seg_path = '/'.join([data_path, sample_dialog["target_image_semantic"]])

    tgt_img_seg = cv2.imread(tgt_img_seg_path)[:512,:512,:]
    tgt_img_seg = cv2.cvtColor(tgt_img_seg, cv2.COLOR_BGR2RGB)
    #print(tgt_img_seg.shape)
    if tgt_img_seg.shape[0] < 512:
    	tgt_img_seg = drawer.post_processing_im(tgt_img_seg)
    	print(tgt_img_seg.shape)
    tgt_img = cv2.imread(tgt_img_path)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

    teller.reset_teller()
    drawer.reset_drawer()

    max_len = 10

    teller.import_tgt_img(tgt_img)
    for i in range(max_len):
        if i == 0:
            teller_utt, terminate_conv= teller.generate_utt()
        else:
            teller_utt, terminate_conv = teller.generate_utt(np.array(drawer_img_real), drawer_utt)
        if not terminate_conv:
            drawer_img_seg = drawer.generate_im(teller_utt)
            drawer_utt = random.choice(drawer.default_drawer_utt)
            current_dialog[sample_dialog_id]["dialog"].append({"teller": teller_utt, "drawer": drawer_utt})
            #get the real img
            drawer_img_seg_obj = Image.fromarray(np.uint8(drawer_img_seg))
            drawer_img_seg_byte = img_to_bytes(drawer_img_seg_obj)
            drawer_img_real, success = segmap_to_real(drawer_img_seg_byte)
            #save the image
            drawer_img_real.save('/'.join([current_output_path,"{}.jpg".format(i)]))
            #save the  seg
            drawer_img_seg_obj.save('/'.join([current_output_path,"{}_seg.png".format(i)]))
        else:
            break
    #Compute Eval Metrics
    #print(drawer_img_seg.shape)
    #print(tgt_img_seg.shape)
    meanIoU_score = mean_IoU(tgt_img_seg[:,:,0], drawer_img_seg, 182)
    print("meanIoU for {} is: {}".format(sample_dialog_id, meanIoU_score))
    meanIoU_list.append(meanIoU_score)
    current_dialog[sample_dialog_id]['meanIoU'] = meanIoU_score
    #print(meanIoU_score)
    seg_scene_similarity = gaugancodraw_eval_metrics(drawer_img_seg,tgt_img_seg[:,:,0], 182)
    print("seg_similarity_score for {} is: {}".format(sample_dialog_id, seg_scene_similarity))
    seg_similarity_list.append(seg_scene_similarity)
    current_dialog[sample_dialog_id]['seg_similarity'] = seg_scene_similarity
    
    #seg_similarity_list
    #save tgt_img to current_output_path
    cv2.imwrite('/'.join([current_output_path, 'target.jpg']), tgt_img)
    cv2.imwrite('/'.join([current_output_path, 'target_seg.png']), tgt_img_seg)

    #dump dialog to the output_path
    with open('/'.join([current_output_path, "dialog.json"]), "w") as f:
        json.dump(current_dialog, f, indent=4, sort_keys=True)

def teller_drawer_talkative_game(sample_dialog, output_path, teller, drawer, meanIoU_list, seg_similarity_list):
    sample_dialog_id = sample_dialog['target_image'].split('/')[-1][:-4]
    #mkdir
    current_output_path = '/'.join([output_path, sample_dialog_id])
    os.makedirs(current_output_path, exist_ok=True)
    current_dialog = {sample_dialog_id: {"dialog":[], "meanIoU": None, "seg_similarity": None}}

    tgt_img_path = '/'.join([data_path, sample_dialog["target_image"]])
    tgt_img_seg_path = '/'.join([data_path, sample_dialog["target_image_semantic"]])

    tgt_img_seg = cv2.imread(tgt_img_seg_path)[:512,:512,:]
    tgt_img_seg = cv2.cvtColor(tgt_img_seg, cv2.COLOR_BGR2RGB)
    #print(tgt_img_seg.shape)
    if tgt_img_seg.shape[0] < 512:
    	tgt_img_seg = drawer.post_processing_im(tgt_img_seg)
    	print(tgt_img_seg.shape)

    tgt_img = cv2.imread(tgt_img_path)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

    teller.reset_teller()
    drawer.reset_drawer()

    max_len = 10

    teller.import_tgt_img(tgt_img)
    for i in range(max_len):
        if i == 0:
            teller_utt, terminate_conv= teller.generate_utt()
        else:
            teller_utt, terminate_conv = teller.generate_utt(np.array(drawer_img_real), drawer_utt)
        if not terminate_conv:
            drawer_img_seg, drawer_utt = drawer.generate_im_utt(teller_utt)
            #drawer_utt = random.choice(drawer.default_drawer_utt)
            current_dialog[sample_dialog_id]["dialog"].append({"teller": teller_utt, "drawer": drawer_utt})
            #get the real img
            drawer_img_seg_obj = Image.fromarray(np.uint8(drawer_img_seg))
            drawer_img_seg_byte = img_to_bytes(drawer_img_seg_obj)
            drawer_img_real, success = segmap_to_real(drawer_img_seg_byte)
            #save the image
            drawer_img_real.save('/'.join([current_output_path,"{}.jpg".format(i)]))
            #save the  seg
            drawer_img_seg_obj.save('/'.join([current_output_path,"{}_seg.png".format(i)]))
        else:
            break
    #Compute Eval Metrics
    #print(drawer_img_seg.shape)
    #print(tgt_img_seg.shape)
    meanIoU_score = mean_IoU(tgt_img_seg[:,:,0], drawer_img_seg, 182)
    print("meanIoU for {} is: {}".format(sample_dialog_id, meanIoU_score))
    current_dialog[sample_dialog_id]['meanIoU'] = meanIoU_score
    meanIoU_list.append(meanIoU_score)
    #print(meanIoU_score)
    seg_scene_similarity = gaugancodraw_eval_metrics(drawer_img_seg,tgt_img_seg[:,:,0], 182)
    print("seg_similarity_score for {} is: {}".format(sample_dialog_id, seg_scene_similarity))
    current_dialog[sample_dialog_id]['seg_similarity'] = seg_scene_similarity
    seg_similarity_list.append(seg_scene_similarity)
    
    seg_similarity_list
    #save tgt_img to current_output_path
    cv2.imwrite('/'.join([current_output_path, 'target.jpg']), tgt_img)
    cv2.imwrite('/'.join([current_output_path, 'target_seg.png']), tgt_img_seg)

    #dump dialog to the output_path
    with open('/'.join([current_output_path, "dialog.json"]), "w") as f:
        json.dump(current_dialog, f, indent=4, sort_keys=True)


if __name__  == "__main__":
	talkative_drawer_cfg = "example_args/gandraw_drawer_args.json"
	silent_drawer_cfg = "example_args/gandraw_args.json"
	teller_cfg = "example_args/gandraw_teller_args.json"

	cfg_teller = load_cfg(teller_cfg)
	cfg_drawer_silent = load_cfg(silent_drawer_cfg)
	cfg_drawer_talkative = load_cfg(talkative_drawer_cfg)

	#Load the source for evaluation
	data_path = "/home/zmykevin/CoDraw_Gaugan/data/GanDraw/data_full_filtered"
	test_json = '/'.join([data_path, 'test.json'])
	target_img_list = []
	with open(test_json, 'r') as f:
		test_data = json.load(f)

	#load the best teller, silent_drawer and silent_drawer
	teller = GanDraw_Teller(cfg_teller, "logs/gandraw/teller/gandraw_teller_lr0.001", iteration=500)
	drawer_silence = GanDraw_Silent_Drawer(cfg_drawer_silent, "logs/gandraw/finetune/baseline1_filtered/filterNew_generator_lr_0.0002", iteration=1000)
	drawer_talkative = GanDraw_Talkative_Drawer(cfg_drawer_talkative, "logs/gandraw/drawer/gandraw_drawer", iteration=1500)

	print("##################Begin Silent Drawer Evaluation#######################################")
	#Compute Scores for Silence Mode
	output_path_silence = "/home/zmykevin/CoDraw_Gaugan/code/GanDraw/GeNeVA/logs/gandraw/interactive_bot/new_teller_silent_drawer"
	meanIoU_list = []
	seg_similarity_list = []
	tgt_im_list = []
	for sample_dialog in test_data['data']:
	    tgt_im_name = sample_dialog["target_image"]
	    if tgt_im_name not in tgt_im_list:
	        teller_drawer_silent_game(sample_dialog, output_path_silence, teller, drawer_silence, meanIoU_list, seg_similarity_list)
	        tgt_im_list.append(tgt_im_name)
	    
	meanIoU_score = sum(meanIoU_list)/len(meanIoU_list)
	seg_similarity_score = sum(seg_similarity_list)/len(seg_similarity_list)
	print("silence_drawer final meanIoU: {}".format(meanIoU_score))
	print("silence_drawer final_seg_similarity_score: {}".format(seg_similarity_score))

	#load the target image

	print("##################Begin Talkative Drawer Evaluation#######################################")

	output_path_talkative= "/home/zmykevin/CoDraw_Gaugan/code/GanDraw/GeNeVA/logs/gandraw/interactive_bot/new_teller_talkative_drawer"
	meanIoU_list = []
	seg_similarity_list = []
	tgt_im_list = []
	for sample_dialog in test_data['data']:
	    tgt_im_name = sample_dialog["target_image"]
	    if tgt_im_name not in tgt_im_list:
	        teller_drawer_talkative_game(sample_dialog, output_path_talkative, teller, drawer_talkative, meanIoU_list, seg_similarity_list)
	        tgt_im_list.append(tgt_im_name)
	    
	meanIoU_score = sum(meanIoU_list)/len(meanIoU_list)
	seg_similarity_score = sum(seg_similarity_list)/len(seg_similarity_list)
	print("talkative_drawer final meanIoU: {}".format(meanIoU_score))
	print("talkative_drawer final_seg_similarity_score: {}".format(seg_similarity_score))

	
