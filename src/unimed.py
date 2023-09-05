import sparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from dataset import MIMICDataset
from model import UniMed
import wandb

WB = False

tv = sparse.load_npz('../data/X.npz').todense()
tiv = sparse.load_npz('../data/S.npz').todense()

df = pd.read_feather('bert_embedding.feather')
labels_df = pd.read_feather('labels.feather')

tv = tv[labels_df['index']]
tv = tv[:, 0:4]
tiv = tiv[labels_df['index']]

#setting up dictionaries for labels + task embeddings
label_index = {}
for task in labels_df.columns[4:]:
    for option in range(2):
        label = f"{task}:{option}"
        label_index[label] = len(label_index)
label_index['<START>'] = len(label_index)

def metrics(outputs, labels, task):
    if task != 'diagnoses':
        outputs = torch.stack(outputs)
        labels = torch.stack(labels)
        pred_proba = nn.Sigmoid()(outputs).detach().cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        auc = roc_auc_score(labels, pred_proba)
        return auc
    else:
        outputs = torch.stack(outputs).detach().cpu()
        labels = torch.stack(labels).detach().cpu().view(-1, 813)
        pred_proba = outputs.softmax(dim=2).view(-1, 813)
        top_k_preds = pred_proba.topk(10, dim=-1).indices
        top_k_target = labels.gather(1, top_k_preds)
        top_k_sum = top_k_target.sum(dim=-1)
        total_sum = labels.sum(dim=-1)
        recall = (top_k_sum / total_sum).mean()
        return recall

train = MIMICDataset(tiv, tv, df, labels_df, 'train', label_index)
val  = MIMICDataset(tiv, tv, df, labels_df, 'val', label_index)
test = MIMICDataset(tiv, tv, df, labels_df, 'test', label_index)

train_dl = DataLoader(train, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
val_dl = DataLoader(val, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
test_dl = DataLoader(test, batch_size=len(test), shuffle=True, pin_memory=True)

if WB:
    wandb.login()
    wandb.init(project='unimed')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UniMed(tiv_hid_size=64, 
               tiv_out_size=50, 
               tv_hid_size=32, 
               tv_out_size=128, 
               num_layers=2, 
               d_model=64, 
               num_heads=4, 
               lin_size=500, 
               dec_hid_size=816, 
               code_size=2000, 
               hid_size=10, 
               label_index=label_index, 
               device=device)
model = model.float()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
if WB:
    wandb.watch(model)

for epoch in range(50):
    model.train()
    total = 0
    train_losses = []
    shock_pred = []; shock_act = []
    arf_pred = []; arf_act = []
    mort_pred = []; mort_act = []
    code_pred = []; code_act = []

    for batch in train_dl:
        optimizer.zero_grad()
        tiv = batch['tiv'].float().to(device)
        tv  = batch['tv'].float().to(device)
        embedding = batch['embedding'].to(device)
        output = batch['output'].to(device)
        task_index = batch['task_index'].to(device)
        shock_labels = batch['shock_labels'].to(device)
        arf_labels = batch['arf_labels'].to(device)
        mort_labels = batch['mort_labels'].to(device)
        code_labels = batch['code_labels'].to(device)

        shock, arf, mort, codes = model(tiv, tv, embedding, output, task_index)
        loss = nn.BCEWithLogitsLoss()(shock.squeeze(), shock_labels.squeeze().to(torch.float))
        loss += nn.BCEWithLogitsLoss()(arf.squeeze(), arf_labels.squeeze().to(torch.float))
        loss += nn.BCEWithLogitsLoss()(mort.squeeze(), mort_labels.squeeze().to(torch.float))
        loss += nn.CrossEntropyLoss()(codes.softmax(dim=1), code_labels.squeeze().to(torch.float))
        train_losses.append(loss)
        loss.backward()
        optimizer.step()

        shock_pred.append(shock.squeeze())
        arf_pred.append(arf.squeeze())
        mort_pred.append(mort.squeeze())
        code_pred.append(codes.squeeze())

        shock_act.append(shock_labels.to(torch.float))
        arf_act.append(arf_labels.to(torch.float))
        mort_act.append(mort_labels.to(torch.float))
        code_act.append(code_labels.to(torch.float))
    
    train_shock_auc = metrics(shock_pred, shock_act, task='shock')
    train_arf_auc = metrics(arf_pred, arf_act, task='arf')
    train_mort_auc = metrics(mort_pred, mort_act, task='mortality')
    train_code_recall = metrics(code_pred, code_act, task='diagnoses')

    print(f"epoch: {epoch}")
    print(f"train shock auc {train_shock_auc}")
    print(f"train arf auc {train_arf_auc}")
    print(f"train mort auc {train_mort_auc}")
    print(f"train code recall {train_code_recall}")
    if WB:
        wandb.log({"train_shock_auc": train_shock_auc,
               "train_arf_auc": train_arf_auc,
               "train_mort_auc": train_mort_auc,
               "train_code_recall": train_code_recall})
    
    val_losses = []

    shock_pred = []
    arf_pred  = []
    mort_pred = []
    code_pred = []

    shock_act = []
    arf_act  = []
    mort_act = []
    code_act = []

    model.eval()
    best_loss = 1e9
    for batch in val_dl:
        optimizer.zero_grad()
        tiv = batch['tiv'].float().to(device)
        tv  = batch['tv'].float().to(device)
        embedding = batch['embedding'].float().to(device)
        output = batch['output'].float().to(device)
        task_index = batch['task_index'].float().to(device)
        shock_labels = batch['shock_labels'].float().to(device)
        arf_labels = batch['arf_labels'].float().to(device)
        mort_labels = batch['mort_labels'].float().to(device)
        code_labels = batch['code_labels'].float().to(device)

        with torch.no_grad():
            shock, arf, mort, codes = model(tiv, tv, embedding, output, task_index)
            loss = nn.BCEWithLogitsLoss()(shock.squeeze(), shock_labels.to(torch.float))
            loss += nn.BCEWithLogitsLoss()(arf.squeeze(), arf_labels.to(torch.float))
            loss += nn.BCEWithLogitsLoss()(mort.squeeze(), mort_labels.to(torch.float))
            loss += nn.CrossEntropyLoss()(codes.softmax(dim=1), code_labels.to(torch.float))
            val_losses.append(loss)
        if loss < best_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_loss = loss
        shock_pred.append(shock.squeeze())
        arf_pred.append(arf.squeeze())
        mort_pred.append(mort.squeeze())
        code_pred.append(codes.squeeze())

        shock_act.append(shock_labels.to(torch.float))
        arf_act.append(arf_labels.to(torch.float))
        mort_act.append(mort_labels.to(torch.float))
        code_act.append(code_labels.to(torch.float))
     
    val_shock_auc = metrics(shock_pred, shock_act, task='shock')
    val_arf_auc = metrics(arf_pred, arf_act, task='arf')
    val_mort_auc = metrics(mort_pred, mort_act, task='mortality')
    val_code_recall = metrics(code_pred, code_act, task='diagnoses')
    if WB: 
        wandb.log({"val_shock_auc": val_shock_auc,
               "val_arf_auc": val_arf_auc,
               "val_mort_auc": val_mort_auc,
               "val_code_recall": val_code_recall})
    print(f"val shock auc {val_shock_auc}")
    print(f"val arf auc {val_arf_auc}")
    print(f"val mort auc {val_mort_auc}")
    print(f"val code recall {val_code_recall}")


test_accs = 0.0    
total = 0
shock_pred = []
arf_pred  = []
mort_pred = []
code_pred = []

shock_act = []
arf_act  = []
mort_act = []
code_act = []

device = torch.device('cpu') # weird error when on gpu
model = model.load_state_dict(torch.load('best_model.pt'))
model.to(device)
model.decoder.mask = model.decoder.mask.to(device)
for batch in test_dl:
    with torch.no_grad():
        tiv = batch['tiv'].float().to(device)
        tv  = batch['tv'].float().to(device)
        embedding = batch['embedding'].to(device)
        output = batch['output'].float().to(device)
        task_index = batch['task_index'].float().to(device)
        shock_labels = batch['shock_labels'].float().to(device)
        arf_labels = batch['arf_labels'].float().to(device)
        mort_labels = batch['mort_labels'].float().to(device)
        code_labels = batch['code_labels'].float().to(device)
        shock, arf, mort, codes = model(tiv, tv, embedding, output, task_index)
        shock_pred.append(shock.squeeze())
        arf_pred.append(arf.squeeze())
        mort_pred.append(mort.squeeze())
        code_pred.append(codes.squeeze())

    shock_act.append(shock_labels.to(torch.float))
    arf_act.append(arf_labels.to(torch.float))
    mort_act.append(mort_labels.to(torch.float))
    code_act.append(code_labels.to(torch.float))

test_shock_auc = metrics(shock_pred, shock_act, task='shock')
test_arf_auc = metrics(arf_pred, arf_act, task='arf')
test_mort_auc = metrics(mort_pred, mort_act, task='mortality')
test_code_recall = metrics(code_pred, code_act, task='diagnoses')
print(f"test shock auc {test_shock_auc}")
print(f"test arf auc {test_arf_auc}")
print(f"test mort auc {test_mort_auc}")
print(f"test code recall {test_code_recall}")
if WB:
    wandb.log({"test_shock_auc": test_shock_auc,
           "test_arf_auc": test_arf_auc,
           "test_mort_auc": test_mort_auc,
           "test_code_recall": test_code_recall})
    wandb.finish()