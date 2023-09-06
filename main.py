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
valid_path = mp2path[dataset]["valid_path"] if "valid_path" in mp2path[dataset] else mp2path[dataset]["test_path"]

train_skill_path = mp2path[dataset]['train_skill_path']
valid_skill_path = mp2path[dataset]["valid_skill_path"] if "valid_skill_path" in mp2path[dataset] else mp2path[dataset]["test_skill_path"]

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
glo.set_value('dataset', dataset)

###################### training
avg_auc = 0
avg_acc = 0


for now_step in range(5):

    best_acc = 0
    best_auc = 0
    state = {'auc': 0, 'acc': 0, 'loss': 0}

    criterion = nn.BCELoss()
    classify = nn.CrossEntropyLoss()

    model = MKT(skill_max, pro_max, d, state_d, p).to(device)

    ccc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

    one_p = 0

    for epoch in range(120):

        one_p += 1

        train_loss, train_acc, train_auc = run_epoch(classify, train_skill_path, model, optimizer,
                                                     skill_neibor_pro, pro_max, train_path, batch_size,
                                                     True, min_seq, max_seq, criterion, device, grad_clip)
        print(
            f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

        valid_loss, valid_acc, valid_auc = run_epoch(classify, valid_skill_path, model, optimizer, skill_neibor_pro,
                                                  pro_max, valid_path, batch_size, False,
                                                  min_seq, max_seq, criterion, device, grad_clip)

        print(f'epoch: {epoch}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_auc: {valid_auc:.4f}')

        if valid_auc >= best_auc:
            one_p = 0
            best_auc = valid_auc
            best_acc = valid_acc
            torch.save(model.state_dict(), f"./MKT_{dataset}_{now_step}_model.pkl")
            state['auc'] = valid_auc
            state['acc'] = valid_acc
            state['loss'] = valid_loss
            torch.save(state, f'./MKT_{dataset}_{now_step}_state.ckpt')
    #         if one_p >= patience:
    #             break

    print(f'*******************************************************************************')
    print(f'best_acc: {best_acc:.4f}, best_auc: {best_auc:.4f}')
    print(f'*******************************************************************************')

    avg_auc += best_auc
    avg_acc += best_acc

avg_auc = avg_auc / 5
avg_acc = avg_acc / 5
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')