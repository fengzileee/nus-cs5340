import pickle
from pathlib import Path
import numpy as np
from .hmm_base import HMMBase

# reference: https://journals.sagepub.com/doi/full/10.1177/1550147718772541

class HMMMultinomialSecondOrder(HMMBase):
    def __init__(self,
                 state_dim,
                 obs_dim,
                 tol=1e-5,
                 max_iters=200,
                 verbose=False):

        # Initialise the dimensions first, so the overriding functions to
        # initialise the transition/observation/initial distributions have
        # access to the dimensions when called from the base class
        self._state_dim = state_dim
        self._obs_dim = obs_dim

        # Initialise the base class
        super().__init__(tol=tol, max_iters=max_iters, verbose=verbose)


    ######################################
    ### HMM model parameter estimation ###
    ######################################

    # Takes in a collection of sequences of the same length, and learns
    # the HMM model from these sequences.
    def estimate_parameters(self, X):
        N, T = X.shape
        prev_log_prob = 0.0
        curr_log_prob = np.inf
        itr = 0

        if self._verbose:
            print("Estimating HMM model parameters...")

        while abs(prev_log_prob - curr_log_prob) > self._tol and itr < self._max_iters:
            itr += 1

            # E-step
            log_gammas = []
            log_xis = []
            average_seq_log_likelihood = 0.0
            for seq in X:
                forward_lattice = self._forward(seq)
                backward_lattice = self._backward(seq)
                seq_log_prob = self._get_sequence_log_likelihood(forward_lattice)

                seq_log_gamma = forward_lattice + backward_lattice - seq_log_prob
                seq_log_xi = np.zeros((self._state_dim, self._state_dim, self._state_dim, T-2))

                for j in range(self._state_dim):
                    for k in range(self._state_dim):
                        for t in range(T-2):
                            i = seq[t-1]
                            seq_log_xi[i, j, k, t] = forward_lattice[j, t] + \
                                                self._get_transition_model_log_prob(i, j, k) + \
                                                self._get_emission_log_prob(k, seq[t+1]) + \
                                                backward_lattice[j, t+1] - \
                                                seq_log_prob

                log_gammas.append(seq_log_gamma)
                log_xis.append(seq_log_xi)
                average_seq_log_likelihood += seq_log_prob

            log_gammas = np.array(log_gammas)
            log_xis = np.array(log_xis)

            # M-step
            self._update_transition_model(log_xis)
            self._update_prior(log_gammas)
            self._update_emission_model(log_gammas, X)

            # Update counters
            prev_log_prob = curr_log_prob
            curr_log_prob = average_seq_log_likelihood / float(N)
            if self._verbose:
                print(f'Iter {itr}, log-likelihood loss: {curr_log_prob}, delta: {abs(prev_log_prob - curr_log_prob)}')


    #######################################################################
    ### Sequence likelihood estimation, forward and backward algorithms ###
    #######################################################################

    # Compute the likelihood of seeing the sequence x under 
    # the current HMM model
    def get_sequence_likelihood(self, x):
        lattice = self._forward(x)
        log_prob = self._get_sequence_log_likelihood(lattice)
        return np.exp(log_prob)

    # Compute the likelihood of seeing the sequence x under
    # the current HMM model. This method uses the backward
    # algorithm, but since forward/backward are functionally
    # equivalent, the output should be exactly the same
    # as get_sequence_likelihood, up to numerical rounding.
    def get_sequence_likelihood_backward(self, x):
        lattice = self._backward(x)
        log_probs = lattice[:, 0] + self._log_phi + self._get_emission_log_prob_batch(x[0])
        log_prob = self._logsumexp(log_probs)
        return np.exp(log_prob)

    def _get_sequence_log_likelihood(self, forward_lattice):
        return self._logsumexp(forward_lattice[:, -1]) # P(O|lambda), probability of observing sequence given parameters

    # Takes in a sequence of length T and computes the alpha values for that sequence
    def _forward(self, x):
        T = len(x)
        dp_table = np.zeros((self._state_dim, T))
        for state in range(self._state_dim):
            dp_table[state, 0] = self._log_phi[state] + self._get_emission_log_prob(state, x[0])
            dp_table[state, 1] = self._log_phi[state] + self._get_emission_log_prob(state, x[1])
        prev_prev_state = x[0]
        prev_state = x[1]
        for t in range(2, T):
            prev_alphas =  dp_table[:, t-1]            
            for state in range(self._state_dim):
                single_transition_log_probs = prev_alphas + self._get_transition_model_log_prob_batch(prev_prev_state, prev_state, state, dst=True)
                single_transition_log_probs += self._get_emission_log_prob(state, x[t])
                dp_table[state, t] = self._logsumexp(single_transition_log_probs)
        return dp_table

    # Takes in a sequence of length T and computes the beta values for that sequence
    def _backward(self, x):
        T = len(x)
        dp_table = np.zeros((self._state_dim, T))
        last_state = x[T-2]
        last_last_state = x[T-1]
        for t in range(T-3, -1, -1):
            prev_betas = dp_table[:, t+1]
            for state in range(self._state_dim):
                single_transition_log_probs =  (prev_betas 
                    + self._get_transition_model_log_prob_batch(state, last_state, last_last_state, dst=False)
                    + self._get_emission_log_prob_batch(x[t+1]))
                dp_table[state, t] = self._logsumexp(single_transition_log_probs)

        return dp_table


    ########################
    ### Viterbi decoding ###
    ########################

    # Uses the Viterbi algorithm to find the most likely sequence
    # of hidden states that generates the observations. This sequence
    # of states can be used to make predictions about future states
    # or emissions from the Markov chain.
    def decode(self, x):
        T = len(x)
        log_path_probs = np.zeros((self._state_dim, T))
        backpointers = np.zeros((self._state_dim, T), dtype=np.int32)
        log_path_probs[:, 0] = self._log_phi + self._get_emission_log_prob_batch(x[0])

        for t in range(2, T):
            prev_prev_state = x[t-2]
            prev_state = x[t-1]
            log_probs_prev_timestep = log_path_probs[:, t-1]
            for state in range(self._state_dim):
                updated_probs = log_probs_prev_timestep + \
                                self._get_transition_model_log_prob_batch(prev_prev_state, prev_state, state, dst=True)
                updated_probs += self._get_emission_log_prob(state, x[t])
                log_path_probs[state, t] = np.max(updated_probs)
                backpointers[state, t] = np.argmax(updated_probs)

        best_log_path_prob = np.max(log_path_probs[:, -1])
        best_path_pointer = np.argmax(log_path_probs[:, -1])
        
        best_path = [best_path_pointer]
        for t in range(T-1, 0, -1):
            best_path.append(backpointers[best_path[-1], t])

        return best_path[::-1], np.exp(best_log_path_prob)


    ##################
    ### Prediction ###
    ##################

    # Does greedy prediction of the succeeding states, solely by
    # looking for the transition from the current state that 
    # has the highest probability, moving to the next state and
    # selecting the emission from that state with the highest
    # probability
    def predict_greedy(self, x, N_future):
        best_path, best_path_prob = self.decode(x)
        state = best_path[-1]
        emissions = []
        for n in range(N_future):
            state = np.argmax(self._get_transition_model_log_prob_batch(state, dst=False))
            emissions.append(np.argmax(self._log_B[state, :]))
        return emissions

    # Takes a brute force combinatorial approach to predicting the
    # succeeding states. Does this by enumerating all possible
    # combinations of future states, concatenating them to the
    # current observed sequence of emissions and computing the
    # forward probability over all of them. The sequence of future
    # states with the highest observation probability is selected
    # as the prediction.
    def predict_brute_force(self, x, N_future):
        raise Exception("TODO: Implement this function")


    ################################
    ### HMM model initialisation ###
    ################################

    def _initialise_transition_model_rand(self):
        self._log_A = [np.log(np.random.dirichlet([1 for i in range(self._state_dim)])) for i in range(self._state_dim)]
        self._log_A = np.array(self._log_A)

    def _initialise_prior_distribution_rand(self):
        self._log_phi = np.log(np.random.dirichlet([1 for i in range(self._state_dim)]).flatten())

    def _initialise_emission_model_rand(self):
        self._log_B = [np.log(np.random.dirichlet([1 for i in range(self._obs_dim)])) for i in range(self._state_dim)]
        self._log_B = np.array(self._log_B)

    def _initialise_transition_model(self, transition_matrix):
        self._log_A = np.log(transition_matrix)

    def _initialise_prior_distribution(self, prior):
        self._log_phi = np.log(prior)

    def _initialise_emission_model(self, emission_matrix):
        self._log_B = np.log(emission_matrix)

    def initialise_parameters(self, transition_matrix, prior, emission_matrix):
        assert(transition_matrix.shape == (self._state_dim, self._state_dim, self._state_dim))
        assert(len(prior) == self._state_dim)
        assert(emission_matrix.shape == (self._state_dim, self._obs_dim))

        self._initialise_transition_model(transition_matrix)
        self._initialise_prior_distribution(prior)
        self._initialise_emission_model(emission_matrix)


    #####################################
    ### Querying HMM model parameters ###
    #####################################

    # Get transition from i --> j
    def _get_transition_model_log_prob(self, i, j, k):
        return self._log_A[i, j, k]
    
    def _get_prior_log_prob(self, state):
        return self._log_phi[state]

    def _get_emission_log_prob(self, state, obs):
        return self._log_B[state, obs]

    # Get vector of log probabilities from the transition model, corresponding
    # to transitions of ALL states --> idx if dst is True, otherwise the
    # transitions of idx --> ALL states if dst is False
    # get i -> j -> k 
    def _get_transition_model_log_prob_batch(self, prev_prev_idx, prev_idx, idx, dst):
        if dst:
            return self._log_A[prev_prev_idx, prev_idx, idx]
        else:
            return self._log_A[idx, prev_idx, prev_prev_idx]

    # Get the emission probabilities of ALL states on obs
    def _get_emission_log_prob_batch(self, obs):
        return self._log_B[:, obs]


    #####################################
    ### Updating HMM model parameters ###
    #####################################

    # Updates the transition model's log probabilities.
    def _update_transition_model(self, log_xis):
        # log_xis is a 5-d tensor with the dimensions
        # (num seq, num states, num states, num_states, traj len).
        # Each value log_xis[s, i, j, k, t] represents the
        # log transition probability of i --> j --> k at timestep t
        # for the kth sequence. 
        #
        # For each state-state pair, sum the gammas over all timesteps 
        # for all sequences. To transform the sum into a probability
        # measure, normalise by summing over the destination states.
        
        summed_over_timesteps = np.apply_along_axis(self._logsumexp, 4, log_xis) 
        summed_over_sequences = np.apply_along_axis(self._logsumexp, 0, summed_over_timesteps)
        summed_over_dst_state = np.apply_along_axis(self._logsumexp, 2, summed_over_sequences)
        self._log_A = summed_over_sequences - np.expand_dims(summed_over_dst_state, axis=0).T


    # Updates the prior log probabilities. Effectively finding for each state
    # the expected frequency of being in that state at the first timestep in
    # the sequence. Computed as a log probability to avoid numerical
    # underflow and rounding errors.
    def _update_prior(self, log_gammas):
        # Note that log_gammas is a 3-d tensor with dimensions
        # (num seq, num states, traj len). We will extract the
        # gammas for t = 0 first, then sum over all sequences
        # for each state.
        N = log_gammas.shape[0]
        log_gammas_t0 = log_gammas[:, :, 0]
        self._log_phi = np.apply_along_axis(self._logsumexp, 0, log_gammas_t0) - np.log(N)

    # Updates the emission model's log probabilities.
    def _update_emission_model(self, log_gammas, X):

        # log_gammas is a 3-d tensor with the dimensions
        # (num seq, num states, traj len). We obtain the normalising
        # factor by summing over all data, i.e. both num seq and traj len
        N, T = X.shape
        summed_over_timesteps = np.apply_along_axis(self._logsumexp, 2, log_gammas)
        summed_over_sequences = np.apply_along_axis(self._logsumexp, 0, summed_over_timesteps)

        # Compute the log likelihood of the emission distribution as the
        # expected no. of times we encounter emission vk in state i over
        # the entire dataset
        for state in range(self._state_dim):
            for vk in range(self._obs_dim):
                valid_log_probs = []
                for n in range(N):
                    for t in range(T):
                        if X[n, t] == vk:
                            valid_log_probs.append(log_gammas[n, state, t])

                if len(valid_log_probs) == 0:
                    self._log_B[state, vk] = -np.inf
                else:
                    self._log_B[state, vk] = self._logsumexp(np.array(valid_log_probs))
                
        self._log_B = self._log_B - np.expand_dims(summed_over_sequences, axis=0).T