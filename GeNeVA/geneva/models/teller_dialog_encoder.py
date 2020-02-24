import torch
import torch.nn as nn
from torch.autograd import Variable
# Develop the Teller Dialog Encoder


class TellerDialogEncoder(nn.Module):

    def __init__(self, cfg):
        # Initialize the Super Class
        super(TellerDialogEncoder, self).__init__()
        self.cfg = cfg
        self.vocabulary_size = cfg.vocab_size
        self.encoder_dim = cfg.image_feat_dim
        # Define the Embedding
        self.embedding = nn.Embedding(self.vocabulary_size, 512)
        # Define the LSTM Cells
        self.lstm = nn.LSTM(self.encoder_dim, self.encoder_dim,
                            1, batch_first=True, dropout=0)

    def forward(self, input_ids, input_lengths, initial_state=None):
        """
        input_ids: the tensor of the ids for the current turn of the conversation

        """
        #ctx_mask = (input_ids != 3)
        embedded_x = self.embedding(input_ids)
        #assert embedded_x.size()[0] == self.cfg.batch_size, "The first dimension should be the batch_size"

        #embedded_x = torch.nn.utils.rnn.pack_padded_sequence(embedded_x, input_lengths, batch_first=True, enforce_sorted=False)
        sorted_len, fwd_order, bwd_order = self.getSortedOrder(input_lengths)
        sorted_input_ids = embedded_x.index_select(dim=0, index=fwd_order)
        # print(sorted_len)
        for i, l in enumerate(sorted_len):
            if l == 0:
                sorted_len[i] = 1

        packed_input_ids = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_input_ids, lengths=sorted_len, batch_first=True)

        if initial_state is None:
            self.lstm.flatten_parameters()
            hx = None
        else:
            # Sort the hidden state vector
            hx_h = initial_state[0].transpose(
                0, 1).index_select(dim=1, index=fwd_order)
            hx_c = initial_state[1].transpose(
                0, 1).index_select(dim=1, index=fwd_order)
            hx = (hx_h, hx_c)

        self.lstm.flatten_parameters()
        _, (h_n, c_n) = self.lstm(packed_input_ids, hx)

        # Using the top layer of the last hidden state vector as the
        # initialization of decoder
        rnn_output = h_n[-1].index_select(dim=0, index=bwd_order)
        h_n = h_n.index_select(dim=1, index=bwd_order).transpose(0, 1)
        c_n = c_n.index_select(dim=1, index=bwd_order).transpose(0, 1)

        return rnn_output, (h_n, c_n)

    def getSortedOrder(self, lens):
        sortedLen, fwdOrder = torch.sort(
            lens.contiguous().view(-1), dim=0, descending=True)
        _, bwdOrder = torch.sort(fwdOrder)
        if isinstance(sortedLen, Variable):
            sortedLen = sortedLen.data

        sortedLen = sortedLen.cpu().numpy().tolist()
        return sortedLen, fwdOrder, bwdOrder
