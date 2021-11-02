import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import math
import hgtk

#random.seed(1488)
#torch.manual_seed(1488)
#torch.cuda.manual_seed(1488)

graphemes = ['PAD', 'SOS'] + list('abcdefghijklmnopqrstuvwxyz.\'-') + ['EOS']
with open('symbols.txt', 'r', encoding='UTF8') as f:
    phonemes = ['PAD', 'SOS'] + f.read().strip().split('\n') + ['EOS']

g2idx = {g: idx for idx, g in enumerate(graphemes)}
idx2g = {idx: g for idx, g in enumerate(graphemes)}

p2idx = {p: idx for idx, p in enumerate(phonemes)}
idx2p = {idx: p for idx, p in enumerate(phonemes)}

def g2seq(s):
    return [g2idx['SOS']] + [g2idx[i] for i in s if i in g2idx.keys()] + [g2idx['EOS']]
    
def seq2g(s):
    return [idx2g[i] for i in s if idx2g[i]]

def p2seq(s):
    return [p2idx['SOS']] + [p2idx[i] for i in s.split() if i in p2idx.keys()] + [p2idx['EOS']]

def seq2p(s):
    return [idx2p[i] for i in s]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, intoken, outtoken, hidden, enc_layers=3, dec_layers=1, dropout=0):
        super(TransformerModel, self).__init__()
        nhead = hidden//64
        
        self.encoder = nn.Embedding(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(hidden, outtoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output

INPUT_DIM = len(graphemes)
OUTPUT_DIM = len(phonemes)

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(INPUT_DIM, OUTPUT_DIM, hidden=128, enc_layers=3, dec_layers=1).to(device)
model.load_state_dict(torch.load("transliteration.pt"))
#model.to(device)
#model.eval()

#random.seed(1488)
#torch.manual_seed(1488)
#torch.cuda.manual_seed(1488)

def transformer_transliteration(text):
    src = torch.tensor(g2seq(text)).unsqueeze(1).cuda()
    memory = model.transformer.encoder(model.pos_encoder(model.encoder(src)))

    out_indexes = [p2idx['SOS'], ]

    for i in range(50):
        trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

        output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
        out_token = output.argmax(2)[-1].item()
        out_indexes.append(out_token)
        if out_token == p2idx['EOS']:
            break
    
    return hgtk.text.compose(seq2p(out_indexes)[1:-1])
