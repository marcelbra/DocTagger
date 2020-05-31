import numpy as np
import os
from torch import nn
from utils_ner import NerDataset, Split, InputExample

class Predictor:

    def __init__(self, tokenizer, splitter, trainer):

        self.tokenizer = tokenizer
        self.splitter = splitter
        self.trainer = trainer
        self.data = None

        self.label_map = {0: "O", 1: "B"}
        self.cache = ["/home/marcel/Desktop/transformers-master/examples/token-classification/BioBERT_ner/pred_data/cached_pred_BertTokenizer_256.lock",
                      "/home/marcel/Desktop/transformers-master/examples/token-classification/BioBERT_ner/pred_data/cached_pred_BertTokenizer_256"]


    def align_predictions(self, predictions: np.ndarray, label_ids: np.ndarray):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.label_map[label_ids[i][j]])
                    preds_list[i].append(self.label_map[preds[i][j]])
        return preds_list, out_label_list

    def set_data(self, tok_sents):
        examples = []
        for guid, sent in enumerate(tok_sents):
            words = [x + "\n" for x in sent]
            labels = ["O" for x in range(len(sent))]
            examples.append(InputExample(guid=f"pred-{guid}", words=words, labels=labels))

        data = NerDataset(
            tokenizer=self.tokenizer,
            examples=examples,
            labels=["B", "O"],
            model_type="BertForTokenClassification",
            max_seq_length=256,
            mode=Split.pred
        )

        self.data = data

    def predict(self):
        predictions, label_ids, metrics = self.trainer.predict(self.data)
        preds_list, _ = self.align_predictions(predictions, label_ids)
        # remove file cache
        for file in self.cache:
            try:
                os.remove(file)
            except OSError:
                pass
        return preds_list