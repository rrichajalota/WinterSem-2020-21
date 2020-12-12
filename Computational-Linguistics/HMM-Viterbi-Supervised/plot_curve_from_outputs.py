'''
Use this file to plot the learning curve of the tagger i.e. training data size vs accuracy.
Run this file from within the shell script plot_learning_curve.sh because this piece of code assumes that accuracy for
 the runs is already present in outputs/learning_curve/ directory.
'''

import glob
import numpy as np
from helper import read_corpus
import matplotlib.pyplot as plt

#train_corpus = read_corpus()
#all_words = [tup[0] for tup in train_corpus.tagged_words()]
#orig_vocab_size = len(set(all_words))

accuracies = []
smoothing = list(np.arange(0.0, 1.1, 0.2))
#training_data_size = np.multiply([0.2, 0.4, 0.6, 0.8, 1.0], orig_vocab_size)

for file in glob.glob("./outputs/learning_curve/smoothing/*.txt"):
    with open(file, 'r') as f:
        for line in f.readlines():
            if 'Accuracy' in line:
                acc = float(line.strip('Accuracy:').strip())
                accuracies.append(acc)

#plt.plot(training_data_size, accuracies)
plt.plot(smoothing, accuracies)
#plt.xlabel('vocab size of training data', fontsize = 14)
plt.xlabel('smoothing factor for add-K smoothing')
plt.ylabel('accuracy', fontsize = 14)
plt.title('Smoothing vs Accuary plot', fontsize = 14)
plt.grid(True)
plt.savefig("./outputs/smoothing_vs_accuracy_curve.png")
plt.show()

