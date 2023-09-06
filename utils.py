import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.utils.data as data
import glo

class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x

class ggg():
    def __init__(self, path):
        self.path = path

    def readData(self):

        problem_list = []
        ans_list = []
        split_char = ','

        read = open(self.path, 'r')
        for index, line in enumerate(read):
            if index % 3 == 0:
                pass

            elif index % 3 == 1:
                problems = line.strip().split(split_char)
                # 由于列表problems每个元素都是char 需要变为int
                problems = list(map(int, problems))
                problem_list += problems

            elif index % 3 == 2:
                ans = line.strip().split(split_char)
                # 由于列表ans每个元素都是char 需要变为int
                ans = list(map(float, ans))
                ans = [int(x) for x in ans]
                ans_list += ans

        read.close()
        return problem_list, ans_list