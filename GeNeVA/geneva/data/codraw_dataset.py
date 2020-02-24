# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""PyTorch Dataset implementation for CoDraw dataset"""
import h5py
import numpy as np
import torch
import torch.nn as nn

from geneva.utils.config import keys

# Created by Mingyang to load the Dialog of the CoDraw Dataset


class CoDrawDialogDataset(nn.Module):

    def __init__(self, path, cfg, img_size=128, glove_path=None):
        super(CoDrawDialogDataset, self).__init__()
        self.dataset = None
        self.dataset_path = path

        # Preprocess the Raw Images
        with h5py.File(path, 'r') as f:
            self.background = f['background'].value.transpose(2, 0, 1)
            self.background = self.background / 128. - 1
            self.background += np.random.uniform(size=self.background.shape,
                                                 low=0, high=1. / 64)

        self.glove = _parse_glove(keys['glove_path'])
        # update cfg
        cfg.vocab_size = len(self.glove.keys())

        with open(keys['codraw_objects'], 'r') as f:
            self.entities = np.array([l.strip() for l in f])

        self.keys = []
        with h5py.File(path, 'r') as f:
            for i in range(len(list(f.keys())) - 1):
                self.keys.append(f[str(i)]['objects'].shape[0])

        self.keys = np.argsort(np.array(self.keys))[::-1]
        self.blocks_maps = {}
        for i in range(0, len(self.keys) - 1, cfg.batch_size):
            block_key = i // cfg.batch_size
            self.blocks_maps[block_key] = self.keys[i:i + cfg.batch_size]

        self.blocks_keys = np.array(list(self.blocks_maps.keys()))

        # Load the Vocab from the Path
        codraw_vocab_path = cfg.codraw_vocab_path
        iclevr_vocab_path = cfg.iclevr_vocab_path
        with open(codraw_vocab_path, 'r') as f:
            codraw_vocab = f.readlines()
            codraw_vocab = [x.strip().rsplit(' ', 1)[0] for x in codraw_vocab]

        # read i-CLEVR vocabulary
        with open(iclevr_vocab_path, 'r') as f:
            clevr_vocab = f.readlines()
            clevr_vocab = [x.strip().rsplit(' ', 1)[0] for x in clevr_vocab]

        self.vocab = ['<s_start>', '<s_end>', '<unk>',
                      '<pad>', '<d_end>'] + codraw_vocab + clevr_vocab
        self.vocab_size = len(self.vocab)

        # format word2ind ind2word
        self.word2index = {k: v for v, k in enumerate(self.vocab)}
        self.index2word = {v: k for v, k in enumerate(self.vocab)}

        self.cfg = cfg

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
        turns = example['utterences'].value
        objects = example['objects'].value
        scene_id = example['scene_id'].value

        turns_tokenized = [t.split() for t in turns]
        lengths = [len(t) for t in turns_tokenized]

        turns_word_embeddings = np.zeros((len(turns), max(lengths), 300))

        for i, turn in enumerate(turns_tokenized):
            for j, w in enumerate(turn):
                turns_word_embeddings[i, j] = self.glove[w]

        #################Extract the teller turns-added by Mingyang############
        teller_turns_tokenized = self.select_teller_turns(turns_tokenized)
        teller_turns = [" ".join(t) for t in teller_turns_tokenized]
        teller_lengths = [len(t) for t in teller_turns_tokenized]

        # Generate the teller_turns_word_embeddings
        teller_turns_word_embeddings = np.zeros(
            (len(teller_turns), max(teller_lengths), 300))
        for i, t_turn in enumerate(teller_turns_tokenized):
            for j, w in enumerate(t_turn):
                teller_turns_word_embeddings[i, j] = self.glove[w]
        #######################################################################

        #################Construct the changed objects#########################
        #changed_objects = np.zeros(objects.shape, dtype='int64')
        changed_objects = np.zeros(objects.shape, dtype='int64')
        for i in range(objects.shape[0]):
            if i > 0:
                changed_objects[i, :] = objects[i, :] - objects[i - 1, :]
            else:
                changed_objects[i, :] = objects[i, :]
        #######################################################################

        # I think it tries to flipper the 'RGB' channel
        images = images[..., ::-1]
        images = images / 128. - 1
        images += np.random.uniform(size=images.shape, low=0, high=1. / 64)
        images = images.transpose(0, 3, 1, 2)

        # Extract the last image as target image, which will be modi
        target_image = images[-1]
        target_image = np.expand_dims(target_image, axis=0)
        #######################################################################

        ###################Extract the Teller Turn Index and Drawer Turn Index#
        teller_turn_ids, drawer_turn_ids, teller_drawer_turn_ids = self.separate_drawer_teller(
            turns_tokenized)
        teller_id_lengths = [len(t) for t in teller_turn_ids]
        drawer_id_lengths = [len(t) for t in drawer_turn_ids]
        teller_drawer_id_lengths = [len(t) for t in teller_drawer_turn_ids]

        sample = {
            'scene_id': scene_id,
            'image': images,
            'turns': turns,
            'objects': objects,
            'turns_word_embedding': turns_word_embeddings,
            'turn_lengths': lengths,
            'background': self.background,
            'entities': self.entities,
            # Added by Mingyang Zhou
            'teller_turns': teller_turns,
            'teller_turns_lengths': teller_lengths,
            'teller_turns_word_embedding': teller_turns_word_embeddings,
            'changed_objects': changed_objects,
            'target_image': target_image,
            'teller_turn_ids': teller_turn_ids,
            'drawer_turn_ids': drawer_turn_ids,
            'teller_drawer_turn_ids': teller_drawer_turn_ids,
            'teller_id_lengths': teller_id_lengths,
            'drawer_id_lengths': drawer_id_lengths,
            'teller_drawer_id_lengths': teller_drawer_id_lengths
        }

        return sample

    def select_teller_turns(self, turns_tokenized):
        """
        turns_tokenized is a list of list of words
        TODO:
        Later, we can extract the drawer's responses.
        """
        teller_turns_tokenized = []
        for t in turns_tokenized:
            teller_index = [i for i, x in enumerate(t) if x == "<teller>"]
            drawer_index = [i for i, x in enumerate(t) if x == "<drawer>"]

            # These Assumptions has to match
            assert len(teller_index) > 0 or len(
                drawer_index) > 0, "No <teller> nor <drawer> toekn found in the turns: {}".format(t)

            if len(drawer_index) == 0:
                teller_t = t.copy()  # Remove the redundant teller_index in the utterance
            elif len(teller_index) == 0:
                teller_t = ["<teller>"]
            else:
                #print("teller_index is: {}".format(teller_index))
                #print("drawer_index is: {}".format(drawer_index))
                t_i = 0
                d_i = 0
                teller_t = []
                previous_role = None
                while t_i < len(teller_index) and d_i < len(drawer_index):
                    if teller_index[t_i] < drawer_index[d_i]:
                        if previous_role == "teller":
                            teller_t += t[teller_index[t_i - 1]                                          :teller_index[t_i]].copy()
                        previous_role = "teller"
                        t_i += 1
                    else:
                        if previous_role == "teller":
                            teller_t += t[teller_index[t_i - 1]                                          :drawer_index[d_i]].copy()
                        previous_role = "drawer"
                        d_i += 1

                if t_i < len(teller_index):
                    teller_t += t[teller_index[t_i]:].copy()
                elif previous_role == "teller":
                    teller_t += t[teller_index[t_i - 1]                                  :drawer_index[d_i]].copy()
                #print("original turn: {}".format(t))
                #print("extracted teller_t turn: {}".format(teller_t))
                assert teller_t.count("<teller>") == t.count(
                    "<teller>"), "There is a mis-match in the teller turns and original turns in '<teller>' token"
            teller_turns_tokenized.append(teller_t)
        assert len(teller_turns_tokenized) == len(
            turns_tokenized), "We didn't get even number of turns on teller's turns"
        return teller_turns_tokenized

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


class CoDrawDataset(nn.Module):

    def __init__(self, path, cfg, img_size=128, glove_path=None):
        super(CoDrawDataset, self).__init__()
        self.dataset = None
        self.dataset_path = path

        # Preprocess the Raw Images
        with h5py.File(path, 'r') as f:
            self.background = f['background'].value.transpose(2, 0, 1)
            self.background = self.background / 128. - 1
            self.background += np.random.uniform(size=self.background.shape,
                                                 low=0, high=1. / 64)

        self.glove = _parse_glove(keys['glove_path'])

        with open(keys['codraw_objects'], 'r') as f:
            self.entities = np.array([l.strip() for l in f])

        self.keys = []
        with h5py.File(path, 'r') as f:
            for i in range(len(list(f.keys())) - 1):
                self.keys.append(f[str(i)]['objects'].shape[0])

        self.keys = np.argsort(np.array(self.keys))[::-1]
        self.blocks_maps = {}
        for i in range(0, len(self.keys) - 1, cfg.batch_size):
            block_key = i // cfg.batch_size
            self.blocks_maps[block_key] = self.keys[i:i + cfg.batch_size]

        self.blocks_keys = np.array(list(self.blocks_maps.keys()))
        self.cfg = cfg

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
        turns = example['utterences'].value
        objects = example['objects'].value
        scene_id = example['scene_id'].value

        turns_tokenized = [t.split() for t in turns]
        lengths = [len(t) for t in turns_tokenized]

        turns_word_embeddings = np.zeros((len(turns), max(lengths), 300))

        for i, turn in enumerate(turns_tokenized):
            for j, w in enumerate(turn):
                turns_word_embeddings[i, j] = self.glove[w]

        images = images[..., ::-1]
        images = images / 128. - 1
        images += np.random.uniform(size=images.shape, low=0, high=1. / 64)
        images = images.transpose(0, 3, 1, 2)

        sample = {
            'scene_id': scene_id,
            'image': images,
            'turns': turns,
            'objects': objects,
            'turns_word_embedding': turns_word_embeddings,
            'turn_lengths': lengths,
            'background': self.background,
            'entities': self.entities,
        }

        return sample

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

    batch_size = len(batch)
    _, c, h, w = batch[0]['image'].shape

    batch_longest_turns = [max(b['turn_lengths']) for b in batch]
    longest_turn = max(batch_longest_turns)

    stacked_images = np.zeros((batch_size, max_len, c, h, w))
    stacked_turns = np.zeros((batch_size, max_len, longest_turn, 300))
    stacked_turn_lengths = np.zeros((batch_size, max_len))
    stacked_objects = np.zeros((batch_size, max_len, 58))
    turns_text = []
    scene_ids = []

    background = None
    for i, b in enumerate(batch):
        img = b['image']
        turns = b['turns']
        background = b['background']
        entities = b['entities']
        turns_word_embedding = b['turns_word_embedding']
        turns_lengths = b['turn_lengths']

        dialog_length = img.shape[0]
        stacked_images[i, :dialog_length] = img
        stacked_turn_lengths[i, :dialog_length] = np.array(turns_lengths)
        stacked_objects[i, :dialog_length] = b['objects']
        turns_text.append(turns)
        scene_ids.append(b['scene_id'])

        for j, turn in enumerate(turns_word_embedding):
            turn_len = turns_lengths[j]
            stacked_turns[i, j, :turn_len] = turn[:turn_len]

    sample = {
        'scene_id': np.array(scene_ids),
        'image': torch.FloatTensor(stacked_images),
        'turn': np.array(turns_text),
        'turn_word_embedding': torch.FloatTensor(stacked_turns),
        'turn_lengths': torch.LongTensor(stacked_turn_lengths),
        'dialog_length': torch.LongTensor(np.array(dialog_lengths)),
        'background': torch.FloatTensor(background),
        'entities': entities,
        'objects': torch.FloatTensor(stacked_objects),
    }

    return sample

# Copy the collan function


# Copy the collan function
def codrawDialog_collate_data(batch):
    batch = sorted(batch, key=lambda x: len(x['image']), reverse=True)
    dialog_lengths = list(map(lambda x: len(x['image']), batch))
    max_len = max(dialog_lengths)

    batch_size = len(batch)
    _, c, h, w = batch[0]['image'].shape

    batch_longest_turns = [max(b['turn_lengths']) for b in batch]
    longest_turn = max(batch_longest_turns)

    stacked_images = np.zeros((batch_size, max_len, c, h, w))
    # 300 is the word2vec dimension
    stacked_turns = np.zeros((batch_size, max_len, longest_turn, 300))
    stacked_turn_lengths = np.zeros((batch_size, max_len))
    stacked_objects = np.zeros((batch_size, max_len, 58))
    turns_text = []
    scene_ids = []

    # Add the additional stacked information
    batch_longest_teller_turns = [
        max(b['teller_turns_lengths']) for b in batch]
    longest_teller_turn = max(batch_longest_teller_turns)
    stacked_target_images = np.zeros((batch_size, 1, c, h, w))
    stacked_teller_turns = np.zeros(
        (batch_size, max_len, longest_teller_turn, 300))
    stacked_teller_turn_lengths = np.zeros((batch_size, max_len))
    # 58 is the category of the objects, this can be set in cfg.
    stacked_changed_objects = np.zeros((batch_size, max_len, 58))
    teller_turns_text = []

#     teller_id_lengths = b["teller_id_lengths"]
#     drawer_id_lengths = b["drawer_id_lengths"]
    batch_longest_teller_ids = [max(b['teller_id_lengths']) for b in batch]
    longest_teller_id_length = max(batch_longest_teller_ids)
    dialog_id = batch_longest_teller_ids.index(longest_teller_id_length)
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
        entities = b['entities']
        turns_word_embedding = b['turns_word_embedding']
        turns_lengths = b['turn_lengths']

        # Add the additional information
        teller_turns = b['teller_turns']
        teller_turns_lengths = b["teller_turns_lengths"]
        teller_turns_word_embedding = b["teller_turns_word_embedding"]
        changed_objects = b["changed_objects"]
        target_image = b["target_image"]
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
        stacked_teller_turn_lengths[
            i, :dialog_length] = np.array(teller_turns_lengths)
        stacked_changed_objects[i, :dialog_length] = changed_objects
        teller_turns_text.append(teller_turns)
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
        for j, teller_turn in enumerate(teller_turns_word_embedding):
            teller_turn_len = teller_turns_lengths[j]
            stacked_teller_turns[
                i, j, :teller_turn_len] = teller_turn[:teller_turn_len]

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
        'entities': entities,
        'objects': torch.FloatTensor(stacked_objects),
        # Include the additional item
        'target_image': torch.FloatTensor(stacked_target_images),
        'teller_turn': np.array(teller_turns_text),
        'teller_turn_word_embedding': torch.FloatTensor(stacked_teller_turns),
        'teller_turn_lengths': torch.LongTensor(stacked_teller_turn_lengths),
        'changed_objects': torch.FloatTensor(stacked_changed_objects),
        'teller_turn_ids': torch.Tensor(stacked_teller_turn_ids),
        'drawer_turn_ids': torch.Tensor(stacked_drawer_turn_ids),
        'teller_drawer_turn_ids': torch.LongTensor(stacked_teller_drawer_turn_ids),
        'teller_id_lengths': torch.LongTensor(stacked_teller_turn_ids_lengths),
        'drawer_id_lengths': torch.LongTensor(stacked_drawer_turn_ids_lengths),
        'teller_drawer_id_lengths': torch.LongTensor(stacked_teller_drawer_turn_ids_lengths)
    }

    return sample
