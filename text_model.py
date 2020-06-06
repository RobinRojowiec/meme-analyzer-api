"""

IDE: PyCharm
Project: meme-analyzer-api
Author: Robin
Filename: text_model.py
Date: 05.06.2020

"""
import torch
import torch.nn as nn
from bpemb import BPEmb


class PretrainedTextModel:
    def __init__(self, dims=100):
        self.dims = dims
        self.bpemb_en = BPEmb(lang="en", dim=dims, vs=10000)

    def encode(self, text):
        ids = self.bpemb_en.encode_ids(text)
        vectors = self.bpemb_en.vectors[ids]
        tensor = torch.from_numpy(vectors)

        # max pooling
        pooled, _ = torch.max(tensor, dim=0)
        return pooled


def get_distance(v1, v2):
    sim_measure = nn.CosineSimilarity(dim=1, eps=1e-6)
    return sim_measure(v1.unsqueeze(dim=0), v2.unsqueeze(dim=0))[0].item()


if __name__ == '__main__':
    m = PretrainedTextModel()
    v = m.encode("red cat")
    v2 = m.encode("white house")
    print(v.shape, v)
    print(v - v2)
    print(get_distance(v, v2))
