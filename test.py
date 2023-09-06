import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import ggg
from run import run_epoch
from model import MKT
import glo

mp2path = {
    'static11': {
        'ques_skill_path': 'data/static11/static11_ques_skill.csv',
        'train_path': 'data/static11/static11_train_question.txt',
        'test_path': 'data/static11/static11_test_question.txt',
        'train_skill_path': 'data/static11/static11_train_skill.txt',
        'test_skill_path': 'data/static11/static11_test_skill.txt',
        'pre_load_gcn': 'data/static11/static11_ques_skill_gcn_adj.pt',
        'skill_max': 106,
        'pro_unique_skill_neibor': 'data/static11/static11_pro_unique_skill_neibor.npy',
        'pro_unique_skill_matrix': 'data/static11/static11_unique_skill_Q-Q'
    }
}
dataset2reg = {
    'static11': 0.1
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = 'static11'

now_regist_reg = dataset2reg[dataset]
ques_skill_path = mp2path[dataset]['ques_skill_path']

train_path = mp2path[dataset]['train_path']
train_skill_path = mp2path[dataset]['train_skill_path']

test_path = mp2path[dataset]["test_path"]
test_skill_path = mp2path[dataset]["test_skill_path"]

skill_max = mp2path[dataset]['skill_max']
gcn_matrix = torch.load(mp2path[dataset]['pre_load_gcn']).to(device)
gat_matrix = torch.load(mp2path[dataset]['pro_unique_skill_matrix']).to(device).to_dense()
skill_neibor_pro = list(np.load(mp2path[dataset]['pro_unique_skill_neibor'], allow_pickle=True))
pro_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 0])

d = 128
state_d = 128
p = 0.4
learning_rate = 0.002
epochs = 15
batch_size = 80
min_seq = 3
max_seq = 200
grad_clip = 15.0
patience = 30

mp_pro_list, _ = ggg(train_path).readData()
mp_skill_list, _ = ggg(train_skill_path).readData()
pro2skill_index = [0 for _ in range(pro_max)]
for x, y in zip(mp_pro_list, mp_skill_list):
    pro2skill_index[x] = y
pro2skill_index = torch.LongTensor(pro2skill_index).to(device)

glo._init()
glo.set_value('pro2skill_index', pro2skill_index)
glo.set_value('max_seq', max_seq)
glo.set_value('gat_matrix', gat_matrix)
glo.set_value('regist_l2', now_regist_reg)
glo.set_value('state_d', state_d)

###################### training
avg_auc = 0
avg_acc = 0
criterion = nn.BCELoss()
classify = nn.CrossEntropyLoss()

model = MKT(skill_max, pro_max, d, state_d, p).to(device)
model.load_state_dict(torch.load(f'saved_model/static11/MKT_static11.pkl', map_location=torch.device('cpu')))

ccc = 0
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

test_loss, test_acc, test_auc = run_epoch(classify, test_skill_path, model, optimizer, skill_neibor_pro,
                                          pro_max, test_path, batch_size, False,
                                          min_seq, max_seq, criterion, device, grad_clip)

print(f'test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')