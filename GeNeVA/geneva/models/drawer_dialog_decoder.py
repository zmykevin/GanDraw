import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

# Define the Decoder of the Teller


class Attention(nn.Module):

    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


class DrawerDialogDecoder(nn.Module):

    def __init__(self, cfg, tf=False):
        super(DrawerDialogDecoder, self).__init__()
        self.use_tf = tf

        self.vocabulary_size = cfg.vocab_size
        self.encoder_dim = cfg.img_encoder_dim

        self.init_h = nn.Linear(self.encoder_dim, 512)
        self.init_c = nn.Linear(self.encoder_dim, 512)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, self.vocabulary_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(self.encoder_dim)
        self.embedding = nn.Embedding(self.vocabulary_size, 512)
        self.lstm = nn.LSTMCell(512 + self.encoder_dim, 512)
        self.cfg = cfg

    def forward(self, img_features, captions, enc_state=None):
        """
        We can use teacher forcing during training. For reference, refer to
        https://www.deeplearningbook.org/contents/rnn.html
        imgfeatures: The fused image features of ground truth and current turn's image
        captions: The ids of the current_turn

        """
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features, enc_state=enc_state)
        # Identify the longest time span in the caption
        max_timespan = max([len(caption) for caption in captions]) - 1

        prev_words = torch.zeros(batch_size, 1).long().cuda()
        if self.use_tf:
            embedding = self.embedding(
                captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan,
                            self.vocabulary_size).cuda()
        alphas = torch.zeros(batch_size, max_timespan,
                             img_features.size(1)).cuda()
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(
                    1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training or not self.use_tf:
                embedding = self.embedding(
                    output.max(1)[1].reshape(batch_size, 1))
        return preds, alphas

    def get_init_lstm_state(self, img_features, enc_state=None):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        if enc_state is not None:
            h = h + enc_state[0].squeeze()
            c = c + enc_state[1].squeeze()
        return h, c

    def caption(self, img_features, beam_size, enc_state=None):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        prev_words = torch.zeros(beam_size, 1).long(
        ).cuda()  # cuda is added by Mingyang
        sentences = prev_words
        # cuda is added by Mingyang
        top_preds = torch.zeros(beam_size, 1).cuda()
        alphas = torch.ones(beam_size, 1, img_features.size(
            1)).cuda()  # cuda is added by Mingyang

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features, enc_state)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(
                    -1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat(
                (sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[
                               prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(
                next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        # In case there is no accomplished sentences
        if len(completed_sentences_preds) == 0:
            for i in range(beam_size):
                completed_sentences.extend(sentences[[i]].tolist())
                completed_sentences_alphas.extend(alphas[[i]].tolist())
                completed_sentences_preds.extend(top_preds[[i]])

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha
