import torch
import torch.nn as nn
import torch.nn.functional as F
from .aggregator import Aggregator

import torch_geometric.nn as gnn
from torch_geometric.nn import MLP, PointNetConv, fps, radius, knn_interpolate


NUM_PROTEINS = 20
NUM_PROTEINS_MASK = NUM_PROTEINS + 1

class SetAbstraction(nn.Module):
    def __init__(self, ratio, ball_query_radius, nn):
        super().__init__()
        self.ratio = ratio
        self.ball_query_radius = ball_query_radius
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.ball_query_radius,
            batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNetPlusPlus(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(
            NUM_PROTEINS_MASK + 1, embed_dim, padding_idx= NUM_PROTEINS_MASK)

        self.sa1_module = SetAbstraction(
            0.5, 0.2,
            MLP([embed_dim + 3, embed_dim, embed_dim, embed_dim * 2])
        )
        self.sa2_module = SetAbstraction(
            0.5, 0.4,
            MLP([embed_dim * 2 + 3, embed_dim * 2, embed_dim * 2, embed_dim * 2])
        )

        self.fp2_module = FPModule(3,
            MLP([embed_dim * 2 + embed_dim * 2, embed_dim * 2, embed_dim * 2]))
        self.fp1_module = FPModule(3,
            MLP([embed_dim * 2 + embed_dim, embed_dim * 2, embed_dim * 2, embed_dim]))

        self.liners1 = nn.Linear(128 + 8, 128)
        self.liners2 = nn.Linear(256 + 8, 256)
        self.liners3 = nn.Linear(256 + 8, 256)
        self.liners4 = nn.Linear(256 + 8, 256)

        self.prompt = nn.Embedding(21, 8)

    def forward(self, x, pos, batch,index):
        x = self.embedding(x)


        # prompt = self.prompt(torch.clamp(torch.div(index,50,rounding_mode='floor'),min=0,max=20))
        prompt = self.prompt(torch.clamp(index//50, min=0, max=20))
        expand_matrix = torch.repeat_interleave(torch.arange(len(index)).to('cuda'), index)
        prompt_atom = prompt[expand_matrix]
        x = self.liners1(torch.cat([x, prompt_atom], dim=-1))

        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        prompt_atom = prompt[sa1_out[-1]]
        sa1_x = self.liners2(torch.cat([sa1_out[0], prompt_atom], dim=-1))
        sa1_out = (sa1_x, sa1_out[1], sa1_out[-1])
        sa2_out = self.sa2_module(*sa1_out)
        prompt_atom = prompt[sa2_out[-1]]
        sa2_x = self.liners3(torch.cat([sa2_out[0], prompt_atom], dim=-1))
        sa2_out = (sa2_x, sa2_out[1], sa2_out[-1])
        fp2_out = self.fp2_module(*sa2_out, *sa1_out)
        prompt_atom = prompt[fp2_out[-1]]
        fp2_x = self.liners4(torch.cat([fp2_out[0], prompt_atom], dim=-1))
        fp2_out = (fp2_x, fp2_out[1], fp2_out[-1])
        fp1_out = self.fp1_module(*fp2_out, *sa0_out)

        x, pos, batch = fp1_out
        return x


class PointNetPlusPlus_encoder(nn.Module):
    def __init__(self, embed_dim=64, global_pool='max'):
        super().__init__()
        self.encoder = PointNetPlusPlus(embed_dim=embed_dim)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'max':
            self.pooling = gnn.global_max_pool
        elif global_pool is None:
            self.pooling = None


    def forward(self, data, other_data=None):
        x, pos, batch = data.x, data.pos, data.batch
        index = data.ptr[1:] - data.ptr[0:-1]
        output = self.encoder(x, pos, batch,index)
        if self.pooling is not None:
            output = self.pooling(output, data.batch)
        return output

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path)['state_dict'])

