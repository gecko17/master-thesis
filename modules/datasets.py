import pandas as pd
import numpy as np
import torch
import nltk

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer,
                 input_special_tokens: bool = False):
        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        sequences = df['sequence'].to_list()
        x_tokenized = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt")
        x_special_tokenized = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        x_nps = {k: v.numpy() for k, v in x_tokenized.items()}

        x_dec = {
            'input_ids': torch.tensor(np.insert(x_nps['input_ids'], 0, tokenizer.cls_token_id, axis=1)).to(self.device),
            'attention_mask': torch.tensor(np.insert(x_nps['attention_mask'], 0, 1, axis=1)).to(self.device),
            'token_type_ids': torch.tensor(np.insert(x_nps['token_type_ids'], 0, 0, axis=1)).to(self.device)
        }

        y = torch.tensor(
            x_special_tokenized['input_ids'].numpy()[
                :, 1:]).to(
            self.device)
        if input_special_tokens:
            x_enc = {k: v.to(self.device)
                     for k, v in x_special_tokenized.items()}
        else:
            x_enc = {k: v.to(self.device) for k, v in x_tokenized.items()}

        self.sequences = sequences
        self.X_enc = x_enc
        self.X_dec = x_dec
        self.Y = y

    def __len__(self):
        return len(self.X_enc['input_ids'])

    def __getitem__(self, idx):
        X_enc_item = {k: v[idx] for k, v in self.X_enc.items()}
        X_dec_item = {k: v[idx] for k, v in self.X_dec.items()}
        return [X_enc_item, X_dec_item, self.Y[idx]]


class SentenceClusterDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, df: pd.DataFrame = None,
                 target_size: float = 0.3):
        self.tokenizer = tokenizer
        self.target_size = target_size
        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        if df is not None and not df.empty:
            self.process_data(df)

    def __len__(self):
        return len(self.X_enc)

    def __getitem__(self, idx):
        return [self.X_enc[idx], self.X_dec[idx], self.Y[idx]]

    def process_data(self, df: pd.DataFrame):
        df['sentences'] = df['CONTENT'].apply(
            lambda x: nltk.sent_tokenize(
                x, language='german'))
        df['sentences'] = df['sentences'].apply(
            lambda x: [
                s for s in x if len(
                    s.split(' ')) > 4 and len(
                    s.split(' ')) < 32])

        df_clusters = pd.DataFrame()
        self.sbert = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        with torch.no_grad():
            for row in df['sentences'].to_numpy():
                if len(row) > 0:
                    sent_emb = self.sbert.encode(row)

                    if len(sent_emb) > 0:
                        num_sentences = len(row)
                        num_clusters = int(num_sentences * self.target_size)
                        num_clusters = 1 if num_clusters == 0 else num_clusters
                        clustering_model = KMeans(n_clusters=num_clusters)
                        clustering_model.fit(sent_emb)
                        cluster_assignment = clustering_model.labels_

                        clustered_sentences = [[] for i in range(num_clusters)]
                        for sentence_id, cluster_id in enumerate(
                                cluster_assignment):
                            clustered_sentences[cluster_id].append(
                                row[sentence_id])

                        df_clusters = df_clusters.append(
                            {'clusters': clustered_sentences}, ignore_index=True)

        df_clusters = df_clusters['clusters'].apply(
            lambda x: pd.Series(x)).stack().reset_index(
            drop=True).to_frame('clusters')
        df_clusters = df_clusters[df_clusters['clusters'].str.len() > 0]

        df_clusters['enc_tokens'] = df_clusters['clusters'].apply(
            lambda cluster: {
                'input_ids': torch.tensor(
                    self.tokenizer(
                        cluster,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt")['input_ids'].numpy()[
                        :,
                        1:]).to(
                    self.device)})
        df_clusters['dec_tokens'] = df_clusters['clusters'].apply(
            lambda cluster: {
                'input_ids': torch.tensor(
                    np.insert(
                        self.tokenizer(
                            cluster,
                            padding=True,
                            truncation=True,
                            add_special_tokens=False,
                            return_tensors="pt")['input_ids'].numpy(),
                        0,
                        self.tokenizer.cls_token_id,
                        axis=1)).to(
                    self.device)})
        df_clusters['y_tokens'] = df_clusters['enc_tokens'].apply(
            lambda cluster: cluster['input_ids'])

        #df['enc_tokens'] = df['enc_tokens'].apply(lambda x: [{'input_ids': s['input_ids'].to(self.device)} for s in x])
        #df_clusters['enc_tokens'] = df_clusters['dec_tokens']
        self.df = df_clusters
        self.X_enc = df_clusters['enc_tokens'].to_list()
        self.X_dec = df_clusters['dec_tokens'].to_list()
        self.Y = df_clusters['y_tokens'].to_list()

    @classmethod
    def from_preprocessed(cls, df: pd.DataFrame, tokenizer):
        dataset = cls(tokenizer=tokenizer)
        dataset.df = df
        dataset.X_enc = df['enc_tokens'].to_list()
        dataset.X_dec = df['dec_tokens'].to_list()
        dataset.Y = df['y_tokens'].to_list()
        return dataset


class DocSplitDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, df: pd.DataFrame = None,
                 desired_splits: int = 4):
        self.tokenizer = tokenizer
        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.min_splits = 2
        self.desired_splits = desired_splits
        self.max_str_len = 1024
        self.min_str_len = 128

        if df is not None and not df.empty:
            self.process_data(df)

    def __len__(self):
        return len(self.X_enc)

    def __getitem__(self, idx):
        return [self.X_enc[idx], self.X_dec[idx], self.Y[idx]]

    def process_data(self, df: pd.DataFrame):
        df['sentences'] = df['CONTENT'].apply(
            lambda x: nltk.sent_tokenize(
                x, language='german'))
        df['content_length'] = df['CONTENT'].str.len()
        df = df[df['content_length'] > 256]

        df_splitted = pd.DataFrame()
        for row in df[['sentences', 'content_length']].to_numpy():
            n_splits, split_len = self.calc_n_splits(
                row[1], self.desired_splits)
            sentences: list = row[0]
            splitted = list()
            for i in range(n_splits):
                sub_doc = ''
                while len(sub_doc) < split_len:
                    if (len(sentences) > 0):
                        sub_doc += sentences.pop(0)
                    else:
                        break
                if len(sub_doc) > 0:
                    splitted.append(sub_doc)
            df_splitted = df_splitted.append(
                {'split_doc': splitted}, ignore_index=True)

        df_splitted['enc_tokens'] = df_splitted['split_doc'].apply(
            lambda doc: {
                'input_ids': torch.tensor(
                    self.tokenizer(
                        doc,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt")['input_ids'].numpy()[
                        :,
                        1:]).to(
                    self.device)})
        df_splitted['dec_tokens'] = df_splitted['split_doc'].apply(
            lambda doc: {
                'input_ids': torch.tensor(
                    np.insert(
                        self.tokenizer(
                            doc,
                            padding=True,
                            truncation=True,
                            add_special_tokens=False,
                            max_lenght=510,  # because of no special_tokens
                            return_tensors="pt")['input_ids'].numpy(),
                        0,
                        self.tokenizer.cls_token_id,
                        axis=1)).to(
                    self.device)})
        df_splitted['y_tokens'] = df_splitted['enc_tokens'].apply(
            lambda doc: doc['input_ids'])

        self.df = df_splitted
        self.X_enc = df_splitted['enc_tokens'].to_list()
        self.X_dec = df_splitted['dec_tokens'].to_list()
        self.Y = df_splitted['y_tokens'].to_list()

    def calc_n_splits(self, doc_len: int, n_splits: int):
        split_len = doc_len / n_splits
        if split_len < self.min_str_len:
            if n_splits > self.min_splits:
                n_splits -= 1
                return self.calc_n_splits(doc_len, n_splits)
            else:
                return n_splits, split_len
        elif split_len > self.max_str_len:
            n_splits += 1
            return self.calc_n_splits(doc_len, n_splits)
        else:
            return n_splits, split_len
