import numpy as np
from itertools import chain, combinations


class HMMBase():
    def __init__(self,
                 tol=1e-3,
                 max_iters=200,
                 verbose=False):
        # Make initialisation of the model parameters deterministic, for our sanity
        np.random.seed(0)

        self._initialise_emission_model_rand()
        self._initialise_transition_model_rand()
        self._initialise_prior_distribution_rand()

        # Initialise parameters
        self._tol = tol
        self._max_iters = max_iters
        self._verbose = verbose
        

    ########################
    ### Helper functions ###
    ########################

    def _logsumexp(self, arr):
        c = np.max(arr)
        return c + np.log(np.sum(np.exp(arr - c)))


    ######################################
    ### HMM model parameter estimation ###
    ######################################

    def estimate_parameters(self):
        raise Exception("To be implemented in derived class!")


    ######################################
    ### Sequence likelihood estimation ###
    ######################################

    # Compute the likelihood of seeing the sequence x under 
    # the current HMM model
    def get_sequence_likelihood(self, x):
        raise Exception("To be implemented in derived class!")

    # Compute the likelihood of seeing the sequence x under
    # the current HMM model. This method uses the backward
    # algorithm, but since forward/backward are functionally
    # equivalent, the output should be exactly the same
    # as get_sequence_likelihood, up to numerical rounding.
    def get_sequence_likelihood_backward(self, x):
        raise Exception("To be implemented in derived class!")


    ########################
    ### Viterbi decoding ###
    ########################

    def decode(self, x):
        raise Exception("To be implemented in derived class!")


    ##################
    ### Prediction ###
    ##################

    def predict_greedy(self, x, N_future):
        raise Exception("To be implemented in derived class!")

    def predict_brute_force(self, x, N_future):
        raise Exception("To be implemented in derived class!")


    ################################
    ### HMM model initialisation ###
    ################################

    def _initialise_transition_model_rand(self):
        raise Exception("To be implemented in derived class!")

    def _initialise_prior_distribution_rand(self):
        raise Exception("To be implemented in derived class!")

    def _initialise_emission_model_rand(self):
        raise Exception("To be implemented in derived class!")

    def _initialise_transition_model(self, transition_matrix):
        raise Exception("To be implemented in derived class!")

    def _initialise_prior_distribution(self, prior):
        raise Exception("To be implemented in derived class!")

    def _initialise_emission_model(self, emission_matrix):
        raise Exception("To be implemented in derived class!")

    def initialise_parameters(self, transition_matrix, prior, emission_matrix):
        raise Exception("To be implemented in derived class!")


    #####################################
    ### Querying HMM model parameters ###
    #####################################

    # Get transition from i --> j
    def _get_transition_model_log_prob(self, i, j):
        raise Exception("To be implemented in derived class!")
    
    def _get_prior_log_prob(self, state):
        raise Exception("To be implemented in derived class!")

    def _get_emission_log_prob(self, state, obs):
        raise Exception("To be implemented in derived class!")

    # Get vector of log probabilities from the transition model, corresponding
    # to transitions of ALL states --> idx if dst is True, otherwise the
    # transitions of idx --> ALL states if dst is False
    def _get_transition_model_log_prob_batch(self, idx, dst):
        raise Exception("To be implemented in derived class!")

    # Get the emission probabilities of ALL states on obs
    def _get_emission_log_prob_batch(self, obs):
        raise Exception("To be implemented in derived class!")


    #####################################
    ### Updating HMM model parameters ###
    #####################################

    def _update_transition_model(self, log_xis):
        raise Exception("To be implemented in derived class!")

    def _update_prior_distribution(self, log_gammas):
        raise Exception("To be implemented in derived class!")

    def _update_emission_model(self, log_gammas, X):
        raise Exception("To be implemented in derived class!")
