# DocTagger
Processes CORD-19 dataset and adds entries for protein/gene tags.

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
