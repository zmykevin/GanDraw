import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

from geneva.utils.config import keys


class GanDrawDataset(nn.Module):

    def __init__(self, path, cfg, img_size=128, glove_path=None):
        super(GanDrawDataset, self).__init__()
        self.dataset = None
        self.dataset_path = path

        self.glove = _parse_glove(keys['glove_gandraw_path'])
        # update cfg
        #cfg.vocab_size = len(self.glove.keys())

        self.keys = []
        with h5py.File(path, 'r') as f:
            # print(len(list(f.keys())))
            for i in range(len(list(f.keys())) - 1):
                #assert f[str(i)]['objects'].shape[0] == f[str(i)]['utterences'].shape[0]
                self.keys.append(f[str(i)]['utterences'].shape[0])

        self.keys = np.argsort(np.array(self.keys))[::-1]
        self.blocks_maps = {}
        for i in range(0, len(self.keys) - 1, cfg.batch_size):
            block_key = i // cfg.batch_size
            self.blocks_maps[block_key] = self.keys[i:i + cfg.batch_size]

        self.blocks_keys = np.array(list(self.blocks_maps.keys()))

        # Load the Vocab from the Path
        gandraw_vocab_path = cfg.gandraw_vocab_path
        with open(gandraw_vocab_path, 'r') as f:
            gandraw_vocab = f.readlines()
            gandraw_vocab = [x.strip().rsplit(' ', 1)[0]
                             for x in gandraw_vocab]

        # print(len(gandraw_vocab))
        self.vocab = ['<s_start>', '<s_end>', '<unk>',
                      '<pad>', '<d_end>'] + gandraw_vocab
        self.vocab_size = len(self.vocab)
        # update the vocab_size
        #cfg.vocab_size = len(self.vocab_size)

        # format word2ind ind2word
        self.word2index = {k: v for v, k in enumerate(self.vocab)}
        self.index2word = {v: k for v, k in enumerate(self.vocab)}

        self.cfg = cfg

        # self.image_transform = transforms.Compose([transforms.ToPILImage(),
        #                                            transforms.ToTensor(),
        #                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Preprocess the Raw Back Ground Images
        self.background = cv2.imread(cfg.gandraw_background)
        self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2RGB)
        if img_size < 128:
            self.background = cv2.resize(self.background, (img_size, img_size), interpolation=cv2.INTER_AREA)
        self.background = np.expand_dims(self.background, axis=0)
        # print(self.background.shape)
        self.background = self.process_image(self.background)

        self.gandraw_entities = {
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
            181: {"name": "wood", "index": 20}
        }
        self.gandraw_entities_len = cfg.num_objects

    def __len__(self):
        with h5py.File(self.dataset_path, 'r') as f:
            return len(list(f.keys())) - 1

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, 'r')

        block_index = self.blocks_keys[idx // self.cfg.batch_size]
        sample_index = idx % self.cfg.batch_size

        if sample_index > len(self.blocks_maps[block_index]) - 1:
            sample_index = len(self.blocks_maps[block_index]) - 1

        example = self.dataset[
            str(self.blocks_maps[block_index][sample_index])]
        images = example['images'].value
        # Added for counting objects
        images_semantic = example['images_semantic'].value

        turns = example['utterences'].value
        scene_id = example['scene_id'].value
        target_images = example['target_images'].value
        target_images_segmentation = example[
            'target_images_segmentation'].value
        target_images_path = example['target_images_path'].value
#         objects = example['objects'].value
#         scene_id = example['scene_id'].value

        turns_tokenized = [t.split() for t in turns]
        lengths = [len(t) for t in turns_tokenized]

        turns_word_embeddings = np.zeros((len(turns), max(lengths), 300))

        for i, turn in enumerate(turns_tokenized):
            for j, w in enumerate(turn):
                turns_word_embeddings[i, j] = self.glove[w]

        # Process Images
        images = self.process_image(images)

        # Process Target Images
        target_images = self.process_image(target_images)
        #######################################################################

        ###################Extract the Teller Turn Index and Drawer Turn Index#
        teller_turn_ids, drawer_turn_ids, teller_drawer_turn_ids = self.separate_drawer_teller(
            turns_tokenized)
        teller_id_lengths = [len(t) for t in teller_turn_ids]
        drawer_id_lengths = [len(t) for t in drawer_turn_ids]
        teller_drawer_id_lengths = [len(t) for t in teller_drawer_turn_ids]

        ##################Extract the objects##################################
        objects = np.zeros(
            (images_semantic.shape[0], self.gandraw_entities_len))
        for j in range(images_semantic.shape[0]):
            current_semantic = images_semantic[j]
            unique_labels = list(np.unique(current_semantic))
            # print(unique_labels)
            for l in unique_labels:
                if self.gandraw_entities.get(l, None) is not None:
                    objects[j][self.gandraw_entities[l]["index"]] = 1
            # print(self.objects[j])
        #######################################################################
        sample = {
            'scene_id': scene_id,
            'image': images,
            'turns': turns,
            'objects': objects,
            'turns_word_embedding': turns_word_embeddings,
            'turn_lengths': lengths,
            'background': self.background,
            # Added by Mingyang Zhou
            'target_image': target_images,
            'target_image_segmentation': target_images_segmentation,
            'target_image_path': target_images_path,
            'teller_turn_ids': teller_turn_ids,
            'drawer_turn_ids': drawer_turn_ids,
            'teller_drawer_turn_ids': teller_drawer_turn_ids,
            'teller_id_lengths': teller_id_lengths,
            'drawer_id_lengths': drawer_id_lengths,
            'teller_drawer_id_lengths': teller_drawer_id_lengths
        }

        return sample

    def process_image(self, images):
        result_images = np.zeros_like(
            images.transpose(0, 3, 1, 2), dtype=np.float32)
        for i in range(images.shape[0]):
            current_img = images[i]
            current_processed_img = self.image_transform(current_img)
            current_processed_img = current_processed_img.numpy()
            result_images[i] = current_processed_img

        return result_images

    def separate_drawer_teller(self, turns_tokenized):
        """
        return two list: 
        1. one list contain the list of list of index for teller
        2. one list contain the lsit of list of index for drawer
        teller_turns_index have one more round dialogs than the actual data which include an end token in the end.
        """
        teller_turns_index = []  # initialize with a sentence start token
        drawer_turns_index = []  # initialize with a sentence start token
        teller_drawer_turns_index = []
        for t in turns_tokenized:
            teller_i_turn = [0]
            drawer_i_turn = [0]
            teller_drawer_i_turn = [0]
            current_role = "teller"
            for x in t:
                if x == "<teller>":
                    current_role = "teller"
                elif x == "<drawer>":
                    current_role = "drawer"
                else:
                    if current_role == "teller":
                        teller_i_turn.append(self.word2index[x])
                    else:
                        drawer_i_turn.append(self.word2index[x])
                    # Concatenate dialog history into one list
                    teller_drawer_i_turn.append(self.word2index[x])

            teller_i_turn.append(1)  # Add an end token
            drawer_i_turn.append(1)  # Add an end token in the end
            teller_drawer_i_turn.append(1)  # Add an end token in the end

            teller_turns_index.append(teller_i_turn)
            drawer_turns_index.append(drawer_i_turn)
            teller_drawer_turns_index.append(teller_drawer_i_turn)
        teller_turns_index.append([0, 4, 1])
        assert len(teller_turns_index) - \
            len(drawer_turns_index) == 1, "The teller has one additional turn than drawer"
        return teller_turns_index, drawer_turns_index, teller_drawer_turns_index

    def shuffle(self):
        np.random.shuffle(self.blocks_keys)


def _parse_glove(glove_path):
    glove = {}
    with open(glove_path, 'r') as f:
        for line in f:
            splitline = line.split()
            word = splitline[0]
            embedding = np.array([float(val) for val in splitline[1:]])
            glove[word] = embedding

    return glove


def collate_data(batch):
    batch = sorted(batch, key=lambda x: len(x['image']), reverse=True)
    dialog_lengths = list(map(lambda x: len(x['image']), batch))
    max_len = max(dialog_lengths)
    #print("max_len is: {}".format(max_len))
    batch_size = len(batch)
    _, c, h, w = batch[0]['image'].shape

    batch_longest_turns = [max(b['turn_lengths']) for b in batch]
    longest_turn = max(batch_longest_turns)

    stacked_images = np.zeros((batch_size, max_len, c, h, w))
    # 300 is the word2vec dimension
    stacked_turns = np.zeros((batch_size, max_len, longest_turn, 300))
    stacked_turn_lengths = np.zeros((batch_size, max_len))
    stacked_objects = np.zeros((batch_size, max_len, 21))
    turns_text = []
    scene_ids = []

    # Add the additional stacked information
    stacked_target_images = np.zeros((batch_size, 1, c, h, w))

    batch_longest_teller_ids = [max(b['teller_id_lengths']) for b in batch]
    longest_teller_id_length = max(batch_longest_teller_ids)
    #dialog_id = batch_longest_teller_ids.index(longest_teller_id_length)
    batch_longest_drawer_ids = [max(b['drawer_id_lengths']) for b in batch]
    longest_drawer_id_length = max(batch_longest_drawer_ids)
    batch_longest_teller_drawer_ids = [
        max(b['teller_drawer_id_lengths']) for b in batch]
    longest_teller_drawer_id_length = max(batch_longest_teller_drawer_ids)

    stacked_teller_turn_ids = np.ones(
        (batch_size, max_len + 1, longest_teller_id_length)) * 3  # 3 is the id  of <pad>
    stacked_drawer_turn_ids = np.ones(
        (batch_size, max_len, longest_drawer_id_length)) * 3  # 3 is the id of <pad>
    stacked_teller_drawer_turn_ids = np.ones(
        (batch_size, max_len, longest_teller_drawer_id_length)) * 3

    stacked_teller_turn_ids_lengths = np.zeros((batch_size, max_len + 1))
    stacked_drawer_turn_ids_lengths = np.zeros((batch_size, max_len))
    stacked_teller_drawer_turn_ids_lengths = np.zeros((batch_size, max_len))

    background = None
    for i, b in enumerate(batch):
        img = b['image']
        turns = b['turns']
        background = b['background']
        turns_word_embedding = b['turns_word_embedding']
        turns_lengths = b['turn_lengths']

        # Add the additional information
        target_image = b["target_image"]
        target_image_segmentation = b["target_image_segmentation"]
        target_image_path = b["target_image_path"]

        teller_turn_ids = b["teller_turn_ids"]
        drawer_turn_ids = b["drawer_turn_ids"]
        teller_drawer_turn_ids = b["teller_drawer_turn_ids"]

        teller_id_lengths = b["teller_id_lengths"]
        drawer_id_lengths = b["drawer_id_lengths"]
        teller_drawer_id_lengths = b["teller_drawer_id_lengths"]

        dialog_length = img.shape[0]
        stacked_images[i, :dialog_length] = img
        stacked_turn_lengths[i, :dialog_length] = np.array(turns_lengths)
        stacked_objects[i, :dialog_length] = b['objects']
        turns_text.append(turns)
        scene_ids.append(b['scene_id'])

        # Update the stacked additional information
        stacked_target_images[i] = target_image
        stacked_teller_turn_ids_lengths[
            i, :len(teller_id_lengths)] = np.array(teller_id_lengths)
        stacked_drawer_turn_ids_lengths[
            i, :len(drawer_id_lengths)] = np.array(drawer_id_lengths)
        stacked_teller_drawer_turn_ids_lengths[
            i, :len(teller_drawer_id_lengths)] = np.array(teller_drawer_id_lengths)

        for j, turn in enumerate(turns_word_embedding):
            turn_len = turns_lengths[j]
            stacked_turns[i, j, :turn_len] = turn[:turn_len]

        # Update the stacked teller_turns
        for j, teller_turn_id in enumerate(teller_turn_ids):
            teller_id_len = teller_id_lengths[j]
            stacked_teller_turn_ids[
                i, j, :teller_id_len] = np.array(teller_turn_id)

        for j, drawer_turn_id in enumerate(drawer_turn_ids):
            drawer_id_len = drawer_id_lengths[j]
            stacked_drawer_turn_ids[
                i, j, :drawer_id_len] = np.array(drawer_turn_id)

        for j, teller_drawer_turn_id in enumerate(teller_drawer_turn_ids):
            teller_drawer_id_len = teller_drawer_id_lengths[j]
            stacked_teller_drawer_turn_ids[
                i, j, :teller_drawer_id_len] = np.array(teller_drawer_turn_id)

    sample = {
        'scene_id': np.array(scene_ids),
        'image': torch.FloatTensor(stacked_images),
        'turn': np.array(turns_text),
        'turn_word_embedding': torch.FloatTensor(stacked_turns),
        'turn_lengths': torch.LongTensor(stacked_turn_lengths),
        'dialog_length': torch.LongTensor(np.array(dialog_lengths)),
        'background': torch.FloatTensor(background),
        'objects': torch.FloatTensor(stacked_objects),
        # Include the additional item
        'target_image': torch.FloatTensor(stacked_target_images),
        'target_image_segmentation': target_image_segmentation,
        'teller_turn_ids': torch.Tensor(stacked_teller_turn_ids),
        'drawer_turn_ids': torch.Tensor(stacked_drawer_turn_ids),
        'teller_drawer_turn_ids': torch.LongTensor(stacked_teller_drawer_turn_ids),
        'teller_id_lengths': torch.LongTensor(stacked_teller_turn_ids_lengths),
        'drawer_id_lengths': torch.LongTensor(stacked_drawer_turn_ids_lengths),
        'teller_drawer_id_lengths': torch.LongTensor(stacked_teller_drawer_turn_ids_lengths)
    }

    return sample
