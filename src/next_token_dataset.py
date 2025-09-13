from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence


class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, eos_id, max_len=None):
        self.texts = texts
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.eos_id = eos_id

    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx],
                               add_special_tokens=False,
                               truncation=True, max_length=self.max_len)
        
        if not ids or ids[-1] != self.eos_id:
            ids = ids + [self.eos_id]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        return {"input_ids": x, "labels": y}


def collate_batch(batch):
    IGNORE_INDEX = -100
    pad_id = 0
    xs  = [b["input_ids"] for b in batch]
    ys  = [b["labels"]    for b in batch]
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)

    X = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    Y = pad_sequence(ys, batch_first=True, padding_value=IGNORE_INDEX)  
    for i, L in enumerate(lengths.tolist()):
        cut = int(0.75 * L)
        if cut > 0:
            Y[i, :cut] = IGNORE_INDEX
    return {"input_ids": X, "labels": Y, "lengths": lengths}