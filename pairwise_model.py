import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig
from transformers import AutoTokenizer
import pandas as pd

AUTH_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained('nguyenvulebinh/vi-mrc-base',
                                          use_auth_token=AUTH_TOKEN)
pad_token_id = tokenizer.pad_token_id


class PairwiseModel(nn.Module):
    def __init__(self, model_name, max_length=384, batch_size=16, device="cuda:0"):
        super(PairwiseModel, self).__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
        self.model.to(self.device )
        self.model.eval()
        self.config = AutoConfig.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
        self.fc = nn.Linear(768, 1).to(self.device)

    def forward(self, ids, masks):
        out = self.model(input_ids=ids,
                         attention_mask=masks,
                         output_hidden_states=False).last_hidden_state
        out = out[:, 0]
        outputs = self.fc(out)
        return outputs

    def stage1_ranking(self, question, texts):
        tmp = pd.DataFrame()
        tmp["text"] = [" ".join(x.split()) for x in texts]
        tmp["question"] = question
        valid_dataset = SiameseDatasetStage1(tmp, tokenizer, self.max_length, is_test=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                                  num_workers=0, shuffle=False, pin_memory=True)
        preds = []
        with torch.no_grad():
            bar = enumerate(valid_loader)
            for step, data in bar:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                preds.append(torch.sigmoid(self(ids, masks)).view(-1))
            preds = torch.concat(preds)
        return preds.cpu().numpy()

    def stage2_ranking(self, question, answer, titles, texts):
        tmp = pd.DataFrame()
        tmp["candidate"] = texts
        tmp["question"] = question
        tmp["answer"] = answer
        tmp["title"] = titles
        valid_dataset = SiameseDatasetStage2(tmp, tokenizer, self.max_length, is_test=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                                  num_workers=0, shuffle=False, pin_memory=True)
        preds = []
        with torch.no_grad():
            bar = enumerate(valid_loader)
            for step, data in bar:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                preds.append(torch.sigmoid(self(ids, masks)).view(-1))
            preds = torch.concat(preds)
        return preds.cpu().numpy()


class SiameseDatasetStage1(Dataset):

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.content1 = tokenizer.batch_encode_plus(list(df.question.values), max_length=max_length, truncation=True)[
            "input_ids"]
        self.content2 = tokenizer.batch_encode_plus(list(df.text.values), max_length=max_length, truncation=True)[
            "input_ids"]
        if not self.is_test:
            self.targets = self.df.label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            'ids1': torch.tensor(self.content1[index], dtype=torch.long),
            'ids2': torch.tensor(self.content2[index][1:], dtype=torch.long),
            'target': torch.tensor(0) if self.is_test else torch.tensor(self.targets[index], dtype=torch.float)
        }


class SiameseDatasetStage2(Dataset):

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.df["content1"] = self.df.apply(lambda row: row.question + f" {tokenizer.sep_token} " + row.answer, axis=1)
        self.df["content2"] = self.df.apply(lambda row: row.title + f" {tokenizer.sep_token} " + row.candidate, axis=1)
        self.content1 = tokenizer.batch_encode_plus(list(df.content1.values), max_length=max_length, truncation=True)[
            "input_ids"]
        self.content2 = tokenizer.batch_encode_plus(list(df.content2.values), max_length=max_length, truncation=True)[
            "input_ids"]
        if not self.is_test:
            self.targets = self.df.label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            'ids1': torch.tensor(self.content1[index], dtype=torch.long),
            'ids2': torch.tensor(self.content2[index][1:], dtype=torch.long),
            'target': torch.tensor(0) if self.is_test else torch.tensor(self.targets[index], dtype=torch.float)
        }


def collate_fn(batch):
    ids = [torch.cat([x["ids1"], x["ids2"]]) for x in batch]
    targets = [x["target"] for x in batch]
    max_len = np.max([len(x) for x in ids])
    masks = []
    for i in range(len(ids)):
        if len(ids[i]) < max_len:
            ids[i] = torch.cat((ids[i], torch.tensor([pad_token_id, ] * (max_len - len(ids[i])), dtype=torch.long)))
        masks.append(ids[i] != pad_token_id)
    # print(tokenizer.decode(ids[0]))
    outputs = {
        "ids": torch.vstack(ids),
        "masks": torch.vstack(masks),
        "target": torch.vstack(targets).view(-1)
    }
    return outputs
