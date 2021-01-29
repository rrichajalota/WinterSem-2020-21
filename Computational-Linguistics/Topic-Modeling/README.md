# Latent Dirichlet Allocation for Topic Modeling using Gibbs Sampling

About
============

LDA is a generative probabilistic model for collections of grouped discrete data. Each group is described as a random mixture over a set of latent topics where
each topic is a discrete distribution over the collection’s vocabulary. The key problem in topic modeling is **posterior inference**. This refers to reversing
the defined generative process and learning the posterior distributions of the latent variables in the model given the observed data.  

Gibbs Sampling is one member of a family of algorithms from the Markov Chain Monte Carlo (MCMC) framework. The MCMC algorithms aim to construct
a Markov chain that has the target posterior distribution as its stationary distribution. In other words, after a number of iterations of stepping through the
chain, sampling from the distribution should converge to be close to sampling from the desired posterior. Check out the [William M. Darling tutorial](https://coli-saar.github.io/cl19/materials/darling-lda.pdf) or [Mark Stevyers](https://webfiles.uci.edu/msteyver/publications/SteyversGriffithsLSABookFormatted.pdf) for more. 

In this code repo, a Gibbs sampler, which resamples a topic for each word in the corpus according to the probability distribution in formula [5] of Griffiths
& Steyvers (2004), has been implemented. 

Requirements
============

have been mentioned in `requirements.txt`.

Project file structure
======================
```
├── gibbs_sampler.py
├── preprocess.py
├── requirements.txt
├── README.md
├── project_report.pdf
├── results
│   ├─...
├── data
│   ├── pptweets_chileEarthquake2014_australiaBushfire2013.txt
└── 
```

Datatsets Used
========

[1] Corpus of 2000 movie reviews from Pang & Lee (2004).  
[2] Custom-made tweets dataset from two events of the [TREC-IS Challenge](http://dcs.gla.ac.uk/~richardm/TREC_IS/).


How to run the code?
===========

-  `gibbs_sampler.py`: is the only file that needs to be run.  
Eg. use `python3 gibbs_sampler.py --filepath [FILEPATH]` to run an experiment with the default arguments. 

```
usage: gibbs_sampler.py [-h] [--filepath FILEPATH] [--num_samples NUM_SAMPLES]
                        [--include_first_line] [--alpha ALPHA] [--beta BETA]
                        [--niterations NITERATIONS] [--ntopics NTOPICS]
                        [--show_words SHOW_WORDS] [--non_random_init]

optional arguments:
  -h, --help            show this help message and exit
  --filepath FILEPATH   use this option to provide a file/corpus for topic
                        modeling.By default, samples from second line onwards
                        are considered (assuming line 1 gives header info). To
                        change this behaviour, use --include_first_line.
  --num_samples NUM_SAMPLES
                        use this option to state the number of samples to take
                        from the given file
  --include_first_line  if specified, includes first line of the dataset.
  --alpha ALPHA         specify the hyperparameter for Dirichlet distribution.
                        (default 0.02)
  --beta BETA           specify the hyperparameter for Dirichlet distribution.
                        (default 0.1)
  --niterations NITERATIONS
                        num of iterations to run (default 500)
  --ntopics NTOPICS     num of topics to find (default 20)
  --show_words SHOW_WORDS
                        number of most frequent words to show (default 40)
  --non_random_init     Specify this argument for a non-random word
                        initialization of topics. When specified, same topic
                        is given to all words of the same document.

```
Depending on the size of the dataset and machine, the runtime varies.

**Note**: The entire implmentation has been done using `numpy` and it took about 4 hours to get the results on Movie Reviews dataset on a machine of 16 G RAM and 12 Cores CPU.
The runtimes have been elaborated further for each experiment file under the `results/` folder and also in the project report. 
Except from the main experiment with default parameters, all other experimets have been performed on a machine of 8 GB and 8 cores. 

## Some interesting observations/lessons learnt during the optimization process

1) using `defaultdict(list)` for maintaining topic assignments works faster than using a variable 2D python `list` or `defaultdict(int)`.  The runtime reduced by approximately **4+ secs per iteration** when the variable 2D vector data structure for topic assignments was replaced by `defaultdict(...)`. There was also a reduction in the initialization runtime by the same factor. 
The difference between `defaultdict(int)` and `defaultdict(list)` for topic assignment is not very significant and might also be system dependent (to be checked). But results from the experiments on my machine are as follows: 
```
 For defaultdict[tuple] = int assignment, initialization took 3.42 secs and 33.64 secs/iteration.
 
 For defaultdict[int] = [topic1, topic2,..] assignment, initialization took 3.23 secs and 32.21 secs/ iteration. 
 ```

2) using `np.zeros(shape, dtype=int)` is faster than `np.zeros(shape, dtype=np.int32/np.int16)` and `np.zeros(shape)` is the fastest. There was a significant reduction in initialization runtime from **8+ secs to 3+ secs** and **10+secs/iteration** when `np.zeros(shape, dtype=np.int32/np.int16)` was replaced with `np.zeros(shape, dtype=int)`. With `np.zeros(shape)`, the reduction was of around 1-2 secs/iteration, with no visible reduction in initialization runtime. 

Both of these observations were made on the default settings with 2000 sentences of the movie review corpus. 


## For extra credit:
- The code has been optimized using numpy.   
- Experiment with a different initialization strategy was performed with the default settings and has been compared with the random initialization strategy (see project report).  
- Apart from experimenting with different hyperparamteres on the Movie Reviews dataset, the topic modeling approach was also applied to a custom-made tweets dataset (external corpora) and multiple experiments were performed.  
