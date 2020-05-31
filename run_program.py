import time
import torch
import spacy
from torch.multiprocessing import set_start_method, Process
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from glob import glob
from doc_builder import DocBuilder

cd_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification"
pretrained_dir = cd_dir + "/biobert_v1.1_pubmed"
cache_dir = cd_dir + "/BioBERT_ner/data"
config_dir = cd_dir + "/biobert_v1.1_pubmed/config.json"
model_dir = cd_dir + "/BioBERT_ner/output_3/pytorch_model.bin"
data_dir = cd_dir + "/BioBERT_ner/pred_data"
labels_dir = cd_dir + "/BioBERT_ner/data/labels.txt"
output_dir = cd_dir + "/BioBERT_ner"
path = "/home/marcel/Desktop/transformers-master/examples/token-classification/CORD-19-research-challenge/document_parses/test_collection"
save_path = "/home/marcel/Desktop/transformers-master/examples/token-classification/saved_paths"

def start_process(paths, tokenizer, splitter, trainer, _id):
    doc_builder = DocBuilder(tokenizer, splitter, trainer, _id)
    curr = doc_builder.get_saved_point()
    for i, path in enumerate(paths):
        if i < curr:
            continue
        data, doc = doc_builder.get_doc(path)
        doc_builder.write_doc(path, data, doc)
        curr = doc_builder.save(i)

if __name__ == '__main__':

    # prepare resources
    set_start_method("spawn", force=True)
    amount_prcs = 7
    paths = [x for x in glob(path + '/**/*.json', recursive=True)]
    model = AutoModelForTokenClassification.from_pretrained(pretrained_dir, config=config_dir)
    model.load_state_dict(torch.load(model_dir))
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=output_dir))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    splitter = spacy.load(cd_dir + "/en_core_sci_md-0.2.4/en_core_sci_md/en_core_sci_md-0.2.4" )

    # start tagging with n processes
    start_run = time.perf_counter()
    processes = []
    for i in range(amount_prcs):
        worker_paths = paths[int((i/amount_prcs)*len(paths)) : int(((i+1)/amount_prcs)*len(paths))]
        p = Process(target=start_process, args=(worker_paths, tokenizer, splitter, trainer, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    end_run = time.perf_counter()

    print(f"Finished processing {len(paths)} documents in {round(end_run - start_run, 2)} seconds.")
    print(f"Each document took about {(end_run - start_run) / len(paths)} seconds to process!")
