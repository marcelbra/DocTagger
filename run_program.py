"""
A multiprocessor program which tags genes and proteins in papers of CORD-19 data set.

How to reproduce experiments:

The program uses a pre-trained BioBERT model fine-tuned on GENETAG.
1. Download pre-trained BioBERT from here: https://github.com/dmis-lab/biobert#download
2. Convert BioBERT Model to PyTorch model as explained here: https://github.com/huggingface/transformers/issues/457
3. Download GENETAG dataset from here: https://www2.informatik.hu-berlin.de/~hakenber/links/benchmarks.html
4. Format GENETAG dataset into CoNLL-2003 format as can be seen here: https://www.clips.uantwerpen.be/conll2003/ner/
   Note: Only column 1 and 4 are needed (so the actual word and its tag)
   To ensure high accuracy I only used 'B' and 'O' tags,
   I was not interested in 'I' tag so you might need to format that as well
5. Fine-tune BioBERT using huggingface's transformers library: https://github.com/huggingface/transformers/tree/master/examples/token-classification
6. Download one of SpaCys tokenizers (for sentence tokenization) at https://allenai.github.io/scispacy/
7. Now everything is set and you need to adjust the paths.

IMPORTANT: To run this you need Nvidia CUDA!
Tested on Ubuntu 20.04, RTX 2080, Ryzen 7 3700X this program needs about
- 2250 MB on GPU for parent process and
- 1100 MB on GPU for each subprocess.
- 6% CPU power for each subprocess
"""

import time
import torch
import spacy

from torch.multiprocessing import set_start_method, Process
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from glob import glob

from doc_builder import DocBuilder

pretrained_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification/biobert_v1.1_pubmed"
model_config_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification/biobert_v1.1_pubmed/config.json"
model_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification/BioBERT_ner/output_3/pytorch_model.bin"
output_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification/BioBERT_ner"
data_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification/CORD-19-research-challenge/document_parses/test_collection"
spacy_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification/en_core_sci_md-0.2.4/en_core_sci_md/en_core_sci_md-0.2.4"
save_path_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification/saved_paths/saved_"
def start_process(paths, tokenizer, trainer, _id):
    """Main method to start tagging documents by each process.
    Gets passed a custom paths list to be processed.
    Saves process after each """
    doc_builder = DocBuilder(tokenizer, trainer, _id, spacy_dir, save_path_dir)
    curr = doc_builder.get_saved_point()
    for i, path in enumerate(paths):
        if i < curr:
            continue
        data, doc = doc_builder.get_doc(path)
        doc_builder.write_doc(path, data, doc)
        curr = doc_builder.save(i)

def main():
    """Runs the multiprocessor."""

    # prepare resources
    set_start_method("spawn", force=True)
    amount_prcs = 7
    paths = [x for x in glob(data_dir + '/**/*.json', recursive=True)]
    model = AutoModelForTokenClassification.from_pretrained(pretrained_dir, config=model_config_dir)
    model.load_state_dict(torch.load(model_dir))
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=output_dir))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    splitter = spacy.load(spacy_dir)

    # start tagging with n processes
    start_run = time.perf_counter()
    processes = []
    for i in range(amount_prcs):
        worker_paths = paths[int((i/amount_prcs)*len(paths)) : int(((i+1)/amount_prcs)*len(paths))]
        p = Process(target=start_process, args=(worker_paths, tokenizer, trainer, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    end_run = time.perf_counter()

    print(f"Finished processing {len(paths)} documents in {round(end_run - start_run, 2)} seconds.")
    print(f"Each document took about {(end_run - start_run) / len(paths)} seconds to process!")

if __name__ == '__main__':
    main()
