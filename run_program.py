from doc_builder import DocBuilder
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import time


start_init = time.perf_counter()
path = "/home/marcel/Desktop/transformers-master/examples/token-classification/CORD-19-research-challenge/document_parses/test_collection"
paths = [x for x in glob(path + '/**/*.json', recursive=True)]
workers_amount = 15
trainers_amount = 5

def main(paths, tokenizer, splitter, trainer):
    doc_builder = DocBuilder(tokenizer, splitter, trainer)
    for path in paths:
        data, doc = doc_builder.get_doc(path)
        doc_builder.write_doc(path, data, doc)

cd_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification"
pretrained_dir = cd_dir + "/biobert_v1.1_pubmed"
cache_dir = cd_dir + "/BioBERT_ner/data"
config_dir = cd_dir + "/biobert_v1.1_pubmed/config.json"
model_dir = cd_dir + "/BioBERT_ner/output_3/pytorch_model.bin"
data_dir = cd_dir + "/BioBERT_ner/pred_data"
labels_dir = cd_dir + "/BioBERT_ner/data/labels.txt"
output_dir = cd_dir + "/BioBERT_ner"


trainers, splitters, tokenizers = [], [], []
for i in range(trainers_amount):
    model = AutoModelForTokenClassification.from_pretrained(pretrained_dir, config=config_dir)
    model.load_state_dict(torch.load(model_dir))
    trainers.append(Trainer(model=model, args=TrainingArguments(output_dir=output_dir)))
    tokenizers.append(AutoTokenizer.from_pretrained(pretrained_dir))
    splitters.append(spacy.load(cd_dir + "/en_core_sci_md-0.2.4/en_core_sci_md/en_core_sci_md-0.2.4"))

end_init = time.perf_counter()
start_run = time.perf_counter()

threads = []
with ThreadPoolExecutor(max_workers=workers_amount) as executor:
    for i in range(workers_amount):
        worker_paths = paths[int(((i - 1) / workers_amount) * len(paths)):int((i / workers_amount) * len(paths))]
        num = int(i/(workers_amount/trainers_amount))
        threads.append(executor.submit(main, worker_paths, tokenizers[num], splitters[num], trainers[num]))

end_run = time.perf_counter()

print()
print(f"Used {workers_amount} workers and {trainers_amount} trainers.")
print(f"Initalizing them took {round(end_init-start_init, 2)} seconds.")
print()
print(f"Finished processing {len(paths)} documents in {round(end_run-start_run, 2)} seconds.")
print(f"Each document took about {(end_run-start_run)/len(paths)} seconds to process!")