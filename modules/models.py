import torch.nn.functional as F
import numpy as np
import torch
import json

from math import log
from transformers import BertModel


class ModelConfig():
    def __init__(self, config_dict: dict = None):
        if config_dict:
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, ModelConfig(value))
                else:
                    setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)

    def to_json_str(self):
        return json.dumps(
            self, default=lambda o: getattr(o, '__dict__', str(o)))

    def to_json_dict(self):
        return json.loads(self.to_json_str())


class Encoder(torch.nn.Module):
    def __init__(self, config: ModelConfig = None,
                 bert_model: torch.nn.Module = None):
        super(Encoder, self).__init__()

        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_model = bert_model

        self.config = config

        if not self.config:
            self.config = ModelConfig({'lstm': {}, 'embeddings': {}})

        if not hasattr(self.config, 'max_sequence_length'):
            self.config.max_sequence_length = 64

        if not hasattr(self.config, 'embedding_dim'):
            self.config.hidden_dim = 768

        if not hasattr(self.config, 'vocab_size'):
            self.config.vocab_size = 30000

        if not hasattr(self.config.lstm, 'layers'):
            self.config.lstm.layers = 1

        if not hasattr(self.config.lstm, 'hidde_dim'):
            self.config.hidden_dim = 768

        if not hasattr(self.config.lstm, 'bidirectional'):
            self.config.lstm.bidirectional = False

        if not hasattr(self.config.lstm, 'dropout'):
            self.config.lstm.bidirectional = 0

        if not hasattr(self.config.embeddings, 'trainable'):
            self.config.embeddings.trainable = False

        if not hasattr(self.config.embeddings, 'dropout'):
            self.config.embeddings.dropout = 0

        if not hasattr(self.config.embeddings, 'mode'):
            self.config.embeddings.mode = 'bert_weights_extracted'

        if not hasattr(self.config, 'sequence_encoding'):
            self.config.sequence_encoding = 'lstm'

        # if self.config.sequence_encoding == 'bert_pooled' and self.config.embeddings.mode != 'bert':
        #  raise ValueError('bert_pooled can only be used with bert as embedding mode')

        if self.config.embeddings.mode == 'bert':
            self.embedding_layer = self.bert_model
        else:
            self.embedding_layer = self.bert_model.get_input_embeddings()
            if not self.config.embeddings.trainable:
                self.embedding_layer.weight.requires_grad = False

        self.embedding_dropout = self.bert_model.embeddings.dropout

        # self.lstm = torch.nn.LSTM(input_size=self.config.embedding_dim, num_layers=self.config.lstm.layers, hidden_size=self.config.lstm.hidden_dim, batch_first=True,
        #                            bidirectional=self.config.lstm.bidirectional, dropout=self.config.lstm.dropout)
        #self.lstm_layers = list()
        # intialize the LSTM
        # input shape is (batch_size, seq_length, 768)
        # output shape for h and c is (batch_size, 768)
        for i in range(self.config.lstm.layers):
            if i == 0:
                setattr(self, f'lstm_{i}', torch.nn.LSTM(
                    input_size=self.config.embedding_dim,
                    num_layers=1,
                    hidden_size=self.config.lstm.hidden_dim,
                    batch_first=True,
                    bidirectional=self.config.lstm.bidirectional))
                # self.lstm_layers.append()
            else:
                hidden_dim = self.config.lstm.hidden_dim * \
                    2 if self.config.lstm.bidirectional else self.config.lstm.hidden_dim
                setattr(self, f'lstm_{i}', torch.nn.LSTM(
                    input_size=hidden_dim,
                    num_layers=1,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    bidirectional=False))
                # self.lstm_layers.append()

    def forward(self, x):
        if self.config.sequence_encoding == 'bert_pooled':
            if self.config.embeddings.trainable:
                pooled = self.embedding_layer(**x).pooler_output
            else:
                with torch.no_grad():
                    pooled = self.embedding_layer(**x).pooler_output
            outputs = None
            hn = pooled.reshape(1, pooled.shape[0], pooled.shape[1])
            cn = hn if self.config.sequence_encoding_cell_state == 'bert_pooled' else torch.zeros(
                (1, pooled.shape[0], pooled.shape[1])).to(self.device)
            return None, hn, cn
        if self.config.embeddings.mode == 'bert':
            if self.config.embeddings.trainable:
                word_embeddings = self.embedding_layer(**x).last_hidden_state
            else:
                with torch.no_grad():
                    word_embeddings = self.embedding_layer(
                        **x).last_hidden_state
        else:
            if self.config.embeddings.dropout > 0:
                word_embeddings = self.embedding_dropout(
                    self.embedding_layer(x['input_ids']))
            else:
                word_embeddings = self.embedding_layer(x['input_ids'])

        if self.config.lstm.layers == 1:
            outputs, (hn, cn) = self.lstm_0(word_embeddings)
            # states not yet concatenated, but instead one tensor of
            # hidden_size per direction
            if self.config.lstm.bidirectional:
                hn = torch.cat((hn[0, :, :], hn[1, :, :]), dim=-
                               1).view(1, hn.shape[1], int(hn.shape[2] * 2))
                cn = torch.cat((cn[0, :, :], cn[1, :, :]), dim=-
                               1).view(1, cn.shape[1], int(cn.shape[2] * 2))
        else:
            output_list = list()
            for i in range(self.config.lstm.layers):
                lstm_layer = getattr(self, f'lstm_{i}')
                if i == 0:
                    # outputs already concatenated from both directions for
                    # bidirectional lstm, hence hidden_size*2
                    outputs, (hn, cn) = lstm_layer(word_embeddings)
                elif i == 1:
                    layer_input = output_list[i - 1]
                    outputs, (hn, cn) = lstm_layer(layer_input)
                else:
                    previous = output_list[i - 1]
                    layer_input = previous.add(outputs)
                    outputs, (hn, cn) = lstm_layer(layer_input)

                output_list.append(outputs)

        return outputs, hn, cn


class Decoder(torch.nn.Module):
    def __init__(self, config: ModelConfig = None,
                 bert_model: torch.nn.Module = None):
        super(Decoder, self).__init__()

        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_model = bert_model

        self.config = config

        if not self.config:
            self.config = ModelConfig({'lstm': {}, 'embeddings': {}})

        if not hasattr(self.config, 'max_sequence_length'):
            self.config.max_sequence_length = 64

        if not hasattr(self.config, 'embedding_dim'):
            self.config.hidden_dim = 768

        if not hasattr(self.config, 'vocab_size'):
            self.config.vocab_size = 30000

        if not hasattr(self.config.lstm, 'layers'):
            self.config.lstm.layers = 1

        if not hasattr(self.config.lstm, 'hidde_dim'):
            self.config.hidden_dim = 768

        if not hasattr(self.config.lstm, 'bidirectional'):
            self.config.lstm.bidirectional = False

        if not hasattr(self.config.lstm, 'dropout'):
            self.config.lstm.bidirectional = 0

        if not hasattr(self.config.embeddings, 'trainable'):
            self.config.embeddings.trainable = False

        if not hasattr(self.config.embeddings, 'dropout'):
            self.config.embeddings.dropout = 0

        if not hasattr(self.config.embeddings, 'mode'):
            self.config.embeddings.mode = 'bert_weights_extracted'

        # if not hasattr(self.config, 'sequence_encoding'):
        #  self.config.sequence_encoding = 'lstm'

        if not hasattr(self.config, 'final_dropout'):
            self.config.final_dropout = 0.5

        # if self.config.sequence_encoding == 'bert_pooled' and self.config.embeddings.mode != 'bert':
        #  raise ValueError('bert_pooled can only be used with bert as embedding mode')

        if self.config.embeddings.mode == 'bert':
            self.embedding_layer = self.bert_model
        else:
            self.embedding_layer = self.bert_model.get_input_embeddings()
            if not self.config.embeddings.trainable:
                self.embedding_layer.weight.requires_grad = False

        self.embedding_dropout = self.bert_model.embeddings.dropout

        # intialize the LSTM
        # input shape is (batch_size, seq_length, 768)
        # output shape for h and c is (batch_size, 768)
        # self.lstm = torch.nn.LSTM(input_size=self.config.embedding_dim, num_layers=self.config.lstm.layers, hidden_size=self.config.lstm.hidden_dim,
        #                          batch_first=True, bidirectional=self.config.lstm.bidirectional, dropout=self.config.lstm.dropout)
        #self.lstm_layers = list()
        # intialize the LSTM
        # input shape is (batch_size, seq_length, 768)
        # output shape for h and c is (batch_size, 768)
        for i in range(self.config.lstm.layers):
            if i == 0:
                setattr(self, f'lstm_{i}', torch.nn.LSTM(
                    input_size=self.config.embedding_dim,
                    num_layers=1,
                    hidden_size=self.config.lstm.hidden_dim,
                    batch_first=True,
                    bidirectional=self.config.lstm.bidirectional))
                # self.lstm_layers.append()
            else:
                hidden_dim = self.config.lstm.hidden_dim * \
                    2 if self.config.lstm.bidirectional else self.config.lstm.hidden_dim
                setattr(self, f'lstm_{i}', torch.nn.LSTM(
                    input_size=hidden_dim,
                    num_layers=1,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    bidirectional=False))
                # self.lstm_layers.append()

        self.linear = torch.nn.Linear(
            self.config.lstm.hidden_dim,
            self.config.vocab_size)
        self.dropout = torch.nn.Dropout(
            p=self.config.final_dropout, inplace=False)
        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, h0, c0):
        if self.config.embeddings.mode == 'bert':
            if self.config.embeddings.trainable:
                word_embeddings = self.embedding_layer(**x).last_hidden_state
            else:
                with torch.no_grad():
                    word_embeddings = self.embedding_layer(
                        **x).last_hidden_state
        else:
            if self.config.embeddings.dropout > 0:
                word_embeddings = self.embedding_dropout(
                    self.embedding_layer(x['input_ids']))
            else:
                word_embeddings = self.embedding_layer(x['input_ids'])

        #outputs, (hn, cn) = self.lstm(word_embeddings, (h0, c0))

        if self.config.lstm.layers == 1:
            outputs, (hn, cn) = self.lstm_0(word_embeddings, (h0, c0))
            # states not yet concatenated, but instead one tensor of
            # hidden_size per direction
            if self.config.lstm.bidirectional:
                hn = torch.cat((hn[0, :, :], hn[1, :, :]), dim=-
                               1).view(1, hn.shape[1], int(hn.shape[2] * 2))
                cn = torch.cat((cn[0, :, :], cn[1, :, :]), dim=-
                               1).view(1, cn.shape[1], int(cn.shape[2] * 2))
        else:
            output_list = list()
            for i in range(self.config.lstm.layers):
                lstm_layer = getattr(self, f'lstm_{i}')
                if i == 0:
                    outputs, (hn, cn) = lstm_layer(word_embeddings, (h0, c0))
                elif i == 1:
                    layer_input = output_list[i - 1]
                    outputs, (hn, cn) = lstm_layer(layer_input)
                else:
                    previous = output_list[i - 1]
                    layer_input = previous.add(outputs)
                    outputs, (hn, cn) = lstm_layer(layer_input)

                output_list.append(outputs)

        if self.config.final_dropout > 0:
            dropped = self.dropout(outputs)
            raw_scores = self.linear(dropped)
        else:
            raw_scores = self.linear(outputs)
        #probabilities = self.softmax(self.linear(outputs))

        return raw_scores, hn, cn


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder_config: ModelConfig,
                 decoder_config: ModelConfig, bert_models: tuple = None):
        super(EncoderDecoder, self).__init__()

        if bert_models is None:
            self.bert_model = BertModel.from_pretrained(
                'bert-base-german-cased')
        else:
            self.bert_model_enc = bert_models[0]
            self.bert_model_dec = bert_models[1]
        # self.embedding_layer = self.bert_model.get_input_embeddings() # returns a torch.nn.Embedding Module
        # self.embedding_dropout =
        # self.bert_model.get_submodule('embeddings').get_submodule('dropout')
        # # returns a torch.nn.Dropout Module

        self.encoder = Encoder(
            config=encoder_config,
            bert_model=self.bert_model)
        self.decoder = Decoder(
            config=decoder_config,
            bert_model=self.bert_model)

        # if embed_mode == 'emb':
        #  self.encoder = Encoder(config=encoder_config, bert_model=self.bert_model)
        #self.encoder = Encoder(hidden_dim=1024, num_layers=1, embed_layer=self.embedding_layer, embed_mode=embed_mode, embed_trainable=embed_trainable, seq_enc_mode=seq_enc_mode, bidirectional=False, lstm_dropout=0)
        #  self.decoder = Decoder(hidden_dim=1024, num_layers=1, embed_layer=self.embedding_layer, embed_mode=embed_mode, embed_trainable=embed_trainable, bidirectional=False, lstm_dropout=0, dropout_final_p=0.5, vocab_size=self.bert_model.config.vocab_size)
        # elif embed_mode == 'bert':
        #  self.encoder = Encoder(config=encoder_config, bert_model=self.bert_model)
        #self.encoder = Encoder(hidden_dim=1024, num_layers=1, embed_layer=self.bert_model, embed_mode=embed_mode, embed_trainable=embed_trainable, seq_enc_mode=seq_enc_mode, lstm_dropout=0.5)
        #  self.decoder = Decoder(hidden_dim=1024, num_layers=1, embed_layer=self.bert_model, embed_mode=embed_mode, embed_trainable=embed_trainable, dropout_final_p=0.5, lstm_dropout=0.5, vocab_size=self.bert_model.config.vocab_size)

        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x_enc, x_dec):
        _, hn, cn = self.encoder(x_enc)
        prob, _, _ = self.decoder(x_dec, hn, cn)

        return prob

    def reconstruct(self, x, n_beams: int = 3, return_all_candidates=False):
        with torch.no_grad():
            _, h, c = self.encoder(x)
            batch_size = x['input_ids'].shape[0]
            if batch_size > 1 and n_beams > 0:
                raise ValueError(
                    "Currently beam search only works for one sample")
            return self.generate_from_encoding(
                h, c, n_beams, return_all_candidates)

    def summarize_sequences(self, x):
        with torch.no_grad():
            # x = list of sequences to summarize to one summary sequence
            _, h, c = self.encoder(x)

            # build mean encodings for hidden and cell states as decoder init
            h0 = torch.mean(h, dim=1, keepdim=True)
            c0 = torch.mean(c, dim=1, keepdim=True)

            return self.generate_from_encoding(
                h0, c0, n_beams=3, return_all_candidates=False)

    def generate_from_encoding(
            self, h0, c0, n_beams: int = 3, return_all_candidates=False):
        # beam search
        if n_beams > 0:
            k = n_beams
            in_candidates = [(torch.tensor([[3]]).to(self.device), 0.0)]
            #out_candidates = [(torch.tensor([[]]).to('cuda:0'), 0.0)]
            for i in range(120):
                candidates = list()
                for m, (cand, prob) in enumerate(in_candidates):
                    dec_inp = {
                        'input_ids': cand
                    }

                    out, hn, cn, = self.decoder(dec_inp, h0, c0)
                    probabilities = F.softmax(out, dim=-1)
                    word_probs, pred_word_ids = torch.topk(
                        probabilities[:, -1:], dim=-1, k=k)

                    for word_prob, word_id in zip(
                            word_probs.view(k), pred_word_ids.view(k)):
                        new_prob = prob - log(word_prob)
                        new_cand = torch.cat(
                            (cand, word_id.view(1, 1)), dim=-1)
                        candidates.append((new_cand, new_prob))

                ordered = sorted(candidates, key=lambda tup: tup[1])
                selected = ordered[:k]
                in_candidates = selected

                end_reached = np.zeros(k)
                for n, (words, _) in enumerate(in_candidates):
                    if torch.any(words == 4):
                        end_reached[n] = 1
                    else:
                        end_reached[n] = 0

                if np.all(end_reached):
                    break

            final_probs = list()
            for _, prob in in_candidates:
                final_probs.append(prob)

            best_candidate_i = np.argmin(final_probs)

            return in_candidates if return_all_candidates else in_candidates[best_candidate_i]
            # return in_candidates if return_all_candidates else
            # in_candidates[best_candidate_i]

        # greedy
        else:
            dec_inp_ids = torch.full(
                (h0.shape[1], 1), 3, dtype=torch.int32).to(
                self.device)

            #dec_inp_ids = torch.tensor([[3]]).to(self.device)
            #dec_out_ids = torch.tensor([[]], dtype=torch.int32).to(self.device)

            for i in range(120):
                dec_inp = {
                    'input_ids': dec_inp_ids
                }
                out, hn, cn, = self.decoder(dec_inp, h0, c0)
                probabilities = F.softmax(out, dim=-1)
                pred_word_id = torch.argmax(probabilities[:, -1:], dim=-1)

                if dec_out_ids:
                    dec_out_ids = torch.cat(
                        [dec_out_ids, pred_word_id], axis=1)
                else:
                    dec_out_ids = pred_word_id

                dec_inp_ids = torch.cat([dec_inp_ids, pred_word_id], axis=1)
                #h = hn
                #c = cn

                if pred_word_id[0, 0] == 4:
                    break

            return dec_out_ids
