import torch
from torch.utils.data import Dataset
class MIMICDataset(Dataset):    
    def __init__(self, tiv, tv, notes, labels, partition, label_index):
        self.tiv = tiv
        self.tv = tv
        self.notes = notes
        self.partition = partition
        self.label_index = label_index
        self.labels = labels[labels['partition'] == self.partition]
        self.tasks = torch.tensor([0, 1, 2] + [3] * 813)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels.iloc[idx, :]
        index = labels['level_0']
        note = self.notes.iloc[index, :]['embed']
        tiv = self.tiv[index]
        tv = self.tv[index]
        output = labels['label_vector'].tolist()
        output.insert(0, self.label_index['<START>'])
        output.pop()
        shock_labels = labels[self.labels.columns[4]]
        arf_labels = labels[self.labels.columns[5]]
        mort_labels = labels[self.labels.columns[6]]
        code_labels = labels[self.labels.columns[7:-1]]/(labels[self.labels.columns[4:-1]]).sum()

        return {
            'tiv': torch.tensor(tiv),
            'tv': torch.tensor(tv),
            'embedding': torch.tensor(note),
            'output': torch.tensor(output),
            'task_index': torch.tensor(self.tasks).clone().detach(),
            'shock_labels': torch.tensor(shock_labels),
            'arf_labels': torch.tensor(arf_labels),
            'mort_labels': torch.tensor(mort_labels),
            'code_labels': torch.tensor(code_labels)
        }
