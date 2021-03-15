# Learning Bilingual word-embeddings using Word Alignments

About
============

The goal of this project is to train bilingual word embeddings using the co-occurrence information obtained from word alignments.

Requirements
============

have been mentioned in `requirements.txt`.

The code was tested on Python 3.6 on Ubuntu 18.04 LTS.

Project file structure
======================
```
.
├── analysis.py
├── cross_lingual_emb.py
├── data
│   ├── eval
│   │   ├── custom_en-de-similarity.txt
│   │   ├── cutsom_de_en_analogy.txt
│   │   ├── de_re-rated_Schm280.txt
│   │   └── rg65_EN-DE-similarity.txt
│   └── parallel_corpus
│       └── trimmed_de_en.txt
├── evaluation.py
├── preprocess.py
├── README.md
├── requirements.txt
├── results
│   ├── eval_bilingual_1000.txt
│   ├── eval_bilingual_5000.txt
│   └── visualizations
│       ├── 1000_corpus
│       │   ├── de250_t-sne_plot_.png
│       │   ├── de-en250_t-sne_plot_.png
│       │   ├── en250_t-sne_plot_.png
│       │   └── loss_vs_epoch1000.png
│       └── 5000_corpus
│           ├── de200_t-sne_plot_.png
│           ├── de250_t-sne_plot_.png
│           ├── de-en200_t-sne_plot_.png
│           ├── de-en230_t-sne_plot_.png
│           ├── de-en250_t-sne_plot_.png
│           ├── en200_t-sne_plot_.png
│           └── en250_t-sne_plot_.png
├── word2vec.py
└── word_alignment.py



```

Datatset
========

- EN-DE EuroParl TMX Corpus from [OPUS](https://opus.nlpl.eu/Europarl.php)  
However, the trimmed dataset that has been used is provided under `data/parallel_corpus/`

How to run the code?
===========

[1] Clean the corpus using `python3 prepocess.py`. This creates input file for fastAlign and two separate parallel files 
for monolingual word2vec training under `data/parallel_corpus/`.

[2] Generate word alignments using [fast_align](https://github.com/clab/fast_align). First, clone the repo and then follow the steps below:
```
mkdir aligned_output
sudo apt-get install libgoogle-perftools-dev libsparsehash-dev
cd fast_align
mkdir build
cd build
cmake ..
make
```
From within `fast_align/build/` run, 
```
./fast_align -i ../../data/parallel_corpus/fastAlign_inp_en-de.txt -d -o -v > ../../aligned_output/en-de-fullCorpus.align
```
[3] Run `python3 word-alignment.py` to generate EN-DE alignment matrix.

[4] Generate monolingual word embeddings using `python3 word2vec.py`. This run creates a `temp_files` folder under `./data`.
An example command would be `python3 word2vec.py ./data/parallel_corpus/processed_de.trg de --sample_size 1000 --ctx_window 3 --n_epochs 50 --min_freq 5 --batch_size 500 --show_logs True`
```
usage: word2vec.py [-h] [--sample_size SAMPLE_SIZE] [--ctx_window CTX_WINDOW] [--dims DIMS]
                   [--generate_training_data GENERATE_TRAINING_DATA] [--n_epochs N_EPOCHS]
                   [--lr LR] [--batch_size BATCH_SIZE] [--temp_dir TEMP_DIR] [--min_freq MIN_FREQ]
                   [--show_logs SHOW_LOGS]
                   file lang

positional arguments:
  file                  use this option to provide the file for which embeddings would be generated
  lang                  specify the lang. abbreviation. of the given file

optional arguments:
  -h, --help            show this help message and exit
  --sample_size SAMPLE_SIZE
                        number of sentences to be considered for training.
  --ctx_window CTX_WINDOW
                        size of the context window.
  --dims DIMS           embedding dimension size.
  --generate_training_data GENERATE_TRAINING_DATA
                        set this arg. to False, if training batches have already been created for
                        the given sample size and ctx window. Saves computation time!!
  --n_epochs N_EPOCHS   number of training epochs
  --lr LR               training hyperparameter: learning rate
  --batch_size BATCH_SIZE
                        embedding dimension size.
  --temp_dir TEMP_DIR   directory path to save intermediate files
  --min_freq MIN_FREQ   Minimum number of times a word should appear in the corpus to be considered
                        for training.
  --show_logs SHOW_LOGS
                        if False, logs will be saved. if True, they would be shown on console.

```

[5] Run `cross_lingual_emb.py` to generate bilingual embeddings. 
``` 
python3 cross_lingual_emb.py --target_weights data/temp_files/de/1000_samples/w1.npy --trg_wd2id_emb data/temp_files/de/1000_samples/word2id.pkl

usage: cross_lingual_emb.py [-h] [--align_mat ALIGN_MAT] [--target_weights TARGET_WEIGHTS] [--trg_vocab_wa TRG_VOCAB_WA] [--src_vocab_wa SRC_VOCAB_WA] [--src_wd2id_wa SRC_WD2ID_WA]
                            [--trg_wd2id_emb TRG_WD2ID_EMB] [--trg_vocab_emb TRG_VOCAB_EMB]

optional arguments:
  -h, --help            show this help message and exit
  --align_mat ALIGN_MAT
                        use this option to provide the path for npy object for alignment_matrix
  --target_weights TARGET_WEIGHTS
                        use this option to provide the path for target weights (de) numpy object
  --trg_vocab_wa TRG_VOCAB_WA
                        use this option to provide the path for the target vocabulary of word alignments corpus.
  --src_vocab_wa SRC_VOCAB_WA
                        use this option to provide the path for the source vocabulary of word alignments corpus.
  --src_wd2id_wa SRC_WD2ID_WA
                        use this option to provide the path for the source word2id mapping of word alignments.
  --trg_wd2id_emb TRG_WD2ID_EMB
                        use this option to provide the path for the target word2id mapping of trained word embeddings.
  --trg_vocab_emb TRG_VOCAB_EMB
                        use this option to provide the path for the target word vocab of trained word embeddings.


```

[6] Run `python3 analysis.py` to get t-sne plot. 
```
python3 analysis.py data/temp_files/bilingual_emb_de-en.npy data/temp_files/bilingual-word2id.pkl data/temp_files/bilingual-wordVocab.pkl de-en --sample 250 --lang1_ratio 0.68 

usage: analysis.py [-h] [--sample SAMPLE] [--seed SEED] [--perplexity PERPLEXITY] [--n_iter N_ITER]
                   [--lr LR] [--lang1_ratio LANG1_RATIO]
                   wt_matrix_file word_to_index_file word_vocab_file lang

positional arguments:
  wt_matrix_file        use this option to provide the path for embedding weights
  word_to_index_file    path for word2index file
  word_vocab_file       path for word_vocab file
  lang                  specify the lang. abbreviation. for bilingual use, en-de format

optional arguments:
  -h, --help            show this help message and exit
  --sample SAMPLE       number of data points to be considered for T-SNE. (default 100)
  --seed SEED           random seed for T-SNE and for sampling random values from the weight
                        matrix. (default 30)
  --perplexity PERPLEXITY
                        perplexity value foor T-SNE (5-50). For N samples, keep perplexity ~
                        sqrt(N). (default 15)
  --n_iter N_ITER       number of iterations for training T-SNE. (default 5000)
  --lr LR               learning rate for T-SNE. (default 40)
  --lang1_ratio LANG1_RATIO
                        In case of bilingual embeddings, if len(vocab_lang1) > len(vocab_lang2),
                        set a ratio for lang1-lang2 i.e. value between [0,1] for it. default = 0.5
                        i.e. both languages are equally represented (default 0.5)


```

[7] Run `python3 evaluation.py` to run evaluation on Analogy and Semantic Word Similarity tasks.
```
usage: evaluation.py [-h] [--word_sim WORD_SIM] [--analogy ANALOGY] [--weights WEIGHTS]
                     [--wd2id WD2ID] [--min MIN] [--max MAX]

optional arguments:
  -h, --help           show this help message and exit
  --word_sim WORD_SIM  use this option to provide the path for word similarity evaluation file
  --analogy ANALOGY    use this option to provide the path for analogy test file
  --weights WEIGHTS    use this option to provide the path for embedding weights.
  --wd2id WD2ID        use this option to provide the path for the word2id dict object for the
                       embeddings.
  --min MIN            use this option to provide the minimum value of the range used for human
                       rating in word similarity eval file
  --max MAX            use this option to provide the maximum value of the range used for human
                       rating in word similarity eval file

```


