import os
import csv

import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.datasets import fetch_20newsgroups


class News20Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, cache_data_dir, word_map_path, max_sent_length=150, max_doc_length=40, is_train=True):
        """
        :param cache_data_path: folder where data files are stored
        :param word_map_path: path for vocab dict, used for embedding
        :param max_sent_length: maximum number of words in a sentence
        :param max_doc_length: maximum number of sentences in a document 
        :param is_train: true if TRAIN mode, false if TEST mode
        """
        self.max_sent_length = max_sent_length
        self.max_doc_length = max_doc_length
        self.split = 'train' if is_train else 'test'

        self.data = fetch_20newsgroups(
            data_home=cache_data_dir,
            subset=self.split,
            categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
            shuffle=False,
            remove=('headers', 'footers', 'quotes'))

        self.vocab = pd.read_csv(
            filepath_or_buffer=word_map_path,
            header=None,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            usecols=[0]).values[:50000]
        self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]

    # NOTE MODIFICATION (REFACTOR)
    def transform(self, text):
        # encode document
        doc = [
            [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(text=sent)]
            for sent in sent_tokenize(text=text)]  # if len(sent) > 0
        doc = [sent[:self.max_sent_length] for sent in doc][:self.max_doc_length]
        num_sents = min(len(doc), self.max_doc_length)

        # skip erroneous ones
        if num_sents == 0:
            return None, -1, None

        num_words = [min(len(sent), self.max_sent_length) for sent in doc][:self.max_doc_length]

        return doc, num_sents, num_words

    def __getitem__(self, i):
        label = self.data['target'][i]
        text = self.data['data'][i]

        # NOTE MODIFICATION (REFACTOR)
        doc, num_sents, num_words = self.transform(text)

        if num_sents == -1:
            return None

        return doc, label, num_sents, num_words

    def __len__(self):
        return len(self.data['data'])

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def num_classes(self):
        return 4
        # return len(list(self.data.target_names))


def collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    docs, labels, doc_lengths, sent_lengths = list(zip(*batch))

    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])

    docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length)).long()
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()

    for doc_idx, doc in enumerate(docs):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.LongTensor(sent_lengths[doc_idx])
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx, :sent_length] = torch.LongTensor(sent)

    return docs_tensor, torch.LongTensor(labels), torch.LongTensor(doc_lengths), sent_lengths_tensor
    
