import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import layers
from configure import Config
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import json
from tensorflow import keras
from tensorflow.keras import layers


# convert to embedding vector sequence
def pre_seq(image, patch_size, weight):
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1,-2)
    patch_embedding = patch * weight
    return patch_embedding


if __name__ == "__main__":
    cfg = Config()

    # test pre_seq
    bs, ic, image_h, image_w = 1, 1, 1, 469
    patch_size = 1
    model_dim = 512
    patch_depth = patch_size * patch_size * ic
    seq = torch.randn(bs, ic, image_h, image_w)
    weight = torch.randn(patch_depth, model_dim)
    max_num_token = 470
    num_classes = 4

    label = torch.randint(4, (bs,))

    # embedding sequence
    patch_embedding = pre_seq(seq, patch_size, weight)

    # add CLS token embedding
    cls_token_embedding = torch.randn(bs, 1, model_dim, requires_grad=True)
    token_embedding = torch.cat([cls_token_embedding, patch_embedding], dim=1)

    # add position embedding
    position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
    seq_len = token_embedding.shape[1]
    position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
    token_embedding += position_embedding

    # pass embedding to transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim,nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    encoder_output = transformer_encoder(token_embedding)

    # do classification
    cls_token_output = encoder_output[:, 0, :]
    linear_layer = nn.Linear(model_dim, num_classes)
    logits = linear_layer(cls_token_output)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, label)

    print(loss)