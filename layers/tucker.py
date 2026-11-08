import torch
from torch import nn

class TuckERLayer(nn.Module):
    """通过关系与核心张量构造查询变换，结合归一化和随机失活获得解码表示。"""
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()

        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))

        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)

        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x


