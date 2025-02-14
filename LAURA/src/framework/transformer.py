import torch
import torch.nn as nn

from src.framework.hyperparameters_commonstructure import *

'''#############################################################'''
'''################### Architecture Modules ####################'''
'''#############################################################'''

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_bins, num_layers_encoder, max_sequence_length, num_layers_decoder):
        super(TransformerModel, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = self.generate_positional_encoding(input_size, max_sequence_length)
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=num_heads,
            num_encoder_layers=num_layers_encoder,
            dim_feedforward=hidden_size,
            num_decoder_layers=num_layers_decoder,
        )
        self.fc = nn.Linear(input_size, num_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x + self.positional_encoding[:self.max_sequence_length,:].to(device) #x.size(0), :].to(device)
        #print(f'x = input with pos_enc : {x.size()}')
        x = self.transformer(x, x)
        #print(f'x transf: {x.size()}')
        x = self.fc(x[:, -1, :])  # Taking the output of the last position as the prediction
        #print(f'x fc: {x.size()}')
        x = self.sigmoid(x)
        return x

    def generate_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    