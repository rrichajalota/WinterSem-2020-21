
import numpy as np
from time import time
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def viterbi(states, observations, Initial_prob, Tranisition_prob, Emission_prob, obs_vocab2ind):
    """
    Given a list of observations and a list of possible hidden states, alongwith the Initial, Transition and
    Emission probability matrices, this function assigns each observation to its corresponding hidden state i.e. acts
    as a tagger. In our case, as a POS-tagger!
    :param states: S = [s1, ..., sN] list of unique states i.e. POS tags. N-dimensional
    :param observations: O = [w1, w2,..., wt] list of words in a sentence. T-dimensional
    :param Initial_prob: c_i = P(X_1 = q_i). N-dim. (i.e. = dim. of states)
    :param Tranisition_prob: a_ij = P(X_t+1 = q_j | X_t = q_i). NxN matrix
    :param Emission_prob: b_i(o) = P(Y_t = o | X_t = q_i ). NxT matrix
    :param obs_vocab2ind: dictionary of the words (that occur in observations) in vocab mapped to their index - enables lookup in O(1) time
    :return: most likely hidden state sequence best_hidden_path = [x1,...,xN], speed of computation
    """
    start = time()

    Viterbi_prob = np.zeros((len(states), len(observations)))  # matrix of max probabilities of dim. NxT
    Backptr_mat = np.zeros((len(states), len(observations)))  # matrix contains prev state corresponding to the max prob.
                                                             # of states stored in Viterbi_prob; NxT dim

    # initial run at t = 0 (with initial probabilities)
    for s in range(len(states)):
        #if we encounter an unknown word,
        Em_w_s = Emission_prob[s][obs_vocab2ind[str(observations[0])]] if str(observations[0]) in obs_vocab2ind else 1

        Viterbi_prob[s, 0] = np.multiply(Initial_prob[s], Em_w_s)
        Backptr_mat[s, 0] = -1  # -1 invalid state to mark that prev. state doesn't exist

    # transtion from t = 1,...,T
    for t, y_obs in enumerate(observations):
        if t == 0:  # already processed during initial run
            continue

        # if we encounter an unknown word,
        y_obs_ind = -1 if str(y_obs) not in obs_vocab2ind else obs_vocab2ind[str(y_obs)]

        for s in range(len(states)):
            # apply smoothing -> crude approach
            Em_w_s = 1 if y_obs_ind == -1 else Emission_prob[s][y_obs_ind]

            Viterbi_prob[s, t] = max(np.multiply(np.multiply(Viterbi_prob[:, t - 1], Tranisition_prob[:, s]), Em_w_s)) # viterbi formula
            Backptr_mat[s, t] = np.argmax(Viterbi_prob[s, t])

    best_path_ind = np.ones((len(observations))).astype(int)  # converted to int, because it would be used as an index
    best_hidden_path = [''] * len(observations)  # list of hidden states

    best_path_ind[-1] = np.argmax(Viterbi_prob[:, len(observations) - 1])  # index of the state that gives the max probability of observing 'o' at t=T
    best_hidden_path[-1] = states[best_path_ind[-1]]  # the corresponding hidden state at t=T

    # loop from right to left and fill in the prev states that are marked by the back_pointers at state, t.
    for t in range(len(best_path_ind) - 1, 0, -1):
        best_path_ind[t - 1] = Backptr_mat[best_path_ind[t]][t]
        best_hidden_path[t - 1] = states[best_path_ind[t - 1]]

    #logging.info(f'Time taken to tag {len(observations)} tokens: {time()-start} secs.')
    return (best_hidden_path, time()-start)

if __name__ == '__main__':
    # toy example - Eisner's ice cream
    states = ['h', 'c']  # states
    states2ind = {'h': 0, 'c': 1}
    obs = [3, 1, 3]  # observations
    obs2ind = {'1': 0, '2': 1, '3': 2}
    Im = np.array([0.8, 0.2])  # follows the order of states as in Q
    Tm = np.array([[0.7, 0.3], [0.4, 0.6]])
    Em = np.array([[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]])
    print(viterbi(states, obs, Im, Tm, Em, obs2ind))

