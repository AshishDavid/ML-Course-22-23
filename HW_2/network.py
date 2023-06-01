import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class ThreeInputsNet(nn.Module):
    def __init__(self, num_tokens, num_cat_features, concat_features, hidden_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_embedding = nn.Embedding(num_tokens, embedding_dim=hidden_size)
        self.title_conv_layer = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.full_embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=hidden_size)
        self.full_conv_layer = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.cat_embedding = nn.Embedding(num_cat_features, embedding_dim=hidden_size)
        self.cat_output_layer = nn.Linear(num_cat_features * hidden_size, hidden_size)

        self.intermediate_layer = nn.Linear(in_features=concat_features, out_features=hidden_size * 2)
        self.additional_layer1 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.additional_layer2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, combined_input):
        input_title, input_full, input_cat = combined_input
        input_title = input_title.long()
        input_full = input_full.long()
        input_cat = input_cat.long()

        title_start = self.title_embedding(input_title).permute((0, 2, 1))
        title = F.relu(self.title_conv_layer(title_start))
        title = F.max_pool1d(title, kernel_size=title.size(2)).squeeze(2)

        full_start = self.full_embedding(input_full).permute((0, 2, 1))
        full = F.relu(self.full_conv_layer(full_start))
        full = F.max_pool1d(full, kernel_size=full.size(2)).squeeze(2)

        cat = self.cat_embedding(input_cat)
        cat = self.cat_output_layer(cat.view(cat.size(0), -1))

        combined_features = torch.cat([title, full, cat], dim=1)

        intermediate_output = F.relu(self.intermediate_layer(combined_features))
        additional_output1 = F.relu(self.additional_layer1(intermediate_output))
        additional_output2 = F.relu(self.additional_layer2(additional_output1))
        final_output = self.output_layer(additional_output2)

        return final_output
