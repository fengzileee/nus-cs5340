import numpy as np
from .hmm_base import HMMBase

class HMMMultinomialFirstOrder(HMMBase):
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
                seq_log_xi = np.zeros((self._state_dim, self._state_dim, T-1))
                for i in range(self._state_dim):
                    for j in range(self._state_dim):
                        for t in range(T-1):
                            seq_log_xi[i, j, t] = forward_lattice[i, t] + \
                                                  self._get_transition_model_log_prob(i, j) + \
                                                  self._get_emission_log_prob(j, seq[t+1]) + \
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
        for t in range(1, T):
            prev_alphas =  dp_table[:, t-1]            
            for state in range(self._state_dim):
                single_transition_log_probs = prev_alphas + self._get_transition_model_log_prob_batch(state, dst=True)
                single_transition_log_probs += self._get_emission_log_prob(state, x[t])
                dp_table[state, t] = self._logsumexp(single_transition_log_probs)
        return dp_table

    # Takes in a sequence of length T and computes the beta values for that sequence
    def _backward(self, x):
        T = len(x)
        dp_table = np.zeros((self._state_dim, T))
        for t in range(T-2, -1, -1):
            prev_betas = dp_table[:, t+1]
            for state in range(self._state_dim):
                single_transition_log_probs =  (prev_betas 
                    + self._get_transition_model_log_prob_batch(state, dst=False)
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

        for t in range(1, T):
            log_probs_prev_timestep = log_path_probs[:, t-1]
            for state in range(self._state_dim):
                updated_probs = log_probs_prev_timestep + \
                                self._get_transition_model_log_prob_batch(state, dst=True)
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
        assert(transition_matrix.shape == (self._state_dim, self._state_dim))
        assert(len(prior) == self._state_dim)
        assert(emission_matrix.shape == (self._state_dim, self._obs_dim))

        self._initialise_transition_model(transition_matrix)
        self._initialise_prior_distribution(prior)
        self._initialise_emission_model(emission_matrix)


    #####################################
    ### Querying HMM model parameters ###
    #####################################

    # Get transition from i --> j
    def _get_transition_model_log_prob(self, i, j):
        return self._log_A[i, j]
    
    def _get_prior_log_prob(self, state):
        return self._log_phi[state]

    def _get_emission_log_prob(self, state, obs):
        return self._log_B[state, obs]

    # Get vector of log probabilities from the transition model, corresponding
    # to transitions of ALL states --> idx if dst is True, otherwise the
    # transitions of idx --> ALL states if dst is False
    def _get_transition_model_log_prob_batch(self, idx, dst):
        if dst:
            return self._log_A[:, idx]
        else:
            return self._log_A[idx, :]

    # Get the emission probabilities of ALL states on obs
    def _get_emission_log_prob_batch(self, obs):
        return self._log_B[:, obs]


    #####################################
    ### Updating HMM model parameters ###
    #####################################

    # Updates the transition model's log probabilities.
    def _update_transition_model(self, log_xis):
        # log_xis is a 4-d tensor with the dimensions
        # (num seq, num states, num states, traj len).
        # Each value log_xis[k, i, j, t] represents the
        # log transition probability of i --> j at timestep t
        # for the kth sequence. 
        #
        # For each state-state pair, sum the gammas over all timesteps 
        # for all sequences. To transform the sum into a probability
        # measure, normalise by summing over the destination states.
        summed_over_timesteps = np.apply_along_axis(self._logsumexp, 3, log_xis)
        summed_over_sequences = np.apply_along_axis(self._logsumexp, 0, summed_over_timesteps)
        summed_over_dst_state = np.apply_along_axis(self._logsumexp, 1, summed_over_sequences)
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


if __name__ == "__main__":
    hmm = HMMMultinomialFirstOrder(2, 2, verbose=True)
    
    # Test using a simple scenario. Consider a coin-flipping task
    # where there are 2 states: 1) we are holding an unbiased coin
    # and 2) we are holding a biased coin. Between coin flips, there
    # is a probability p that we switch coins. This gives a Markov chain
    # with a 2x2 transition matrix. The unbiased coin state has index 0
    # and the biased coin state has index 1.
    #
    # There are two possible observations/emissions from each coin flip.
    # We can either observe: 1) heads or 2) tails. The probability of
    # observing heads/tails from the unbiased coin state is 0.5, and
    # the probability from the biased coin state is q/1-q, for some 
    # probability q. The heads emission has index 0 and the tails
    # emission has index 1.

    def make_transition_matrix(p):
        return np.array([[p, 1-p], [1-p, p]])

    def make_emission_matrix(q):
        return np.array([[0.5, 0.5], [q, 1-q]])

    p = 0.5
    q = 1
    transition_matrix = make_transition_matrix(p)
    emission_matrix = make_emission_matrix(q)
    prior = np.array([0.5, 0.5])

    hmm.initialise_parameters(transition_matrix, prior, emission_matrix)
    test_seq1 = np.array([1, 1])
    test_seq1_prob = hmm.get_sequence_likelihood_backward(test_seq1)
    test_seq1_prob2 = hmm.get_sequence_likelihood(test_seq1)
    print(test_seq1_prob, test_seq1_prob2)

    test_seq2 = np.array([1, 0, 0])
    path, prob = hmm.decode(test_seq2)
    print(path)
    print(prob)

    # test_seq1 = np.array([0, 1])
    # test_seq2 = np.array([0, 0])
    # test_seq3 = np.array([1, 1])
    # test_seq4 = np.array([1, 0])
    
    # data = np.array([test_seq1, test_seq2, test_seq3, test_seq4])
    # # data = np.array([test_seq1, test_seq4])
    # hmm.estimate_parameters(data)
    # print(np.exp(hmm._log_A))
    # print(np.exp(hmm._log_B))
    # print(np.exp(hmm._log_phi))

    # def draw_sequence(A, B, prior, T):
    #     num_states, num_obs = B.shape
    #     state = np.random.choice(num_states, 1, p=prior).item()
    #     emissions = [np.random.choice(num_obs, 1, p=B[state].flatten()).item()]
    #     for t in range(1, T):
    #         state = np.random.choice(num_states, 1, p=A[state].flatten()).item()
    #         emissions.append(np.random.choice(num_obs, 1, p=B[state].flatten()).item())
    #     return emissions

    # data = np.array([draw_sequence(transition_matrix, emission_matrix, prior, 10) for i in range(500)])
    # print(data)

    # hmm._initialise_transition_model(transition_matrix)
    # hmm._initialise_prior_distribution(prior)
    # hmm.estimate_parameters(data)
    # print(np.exp(hmm._log_A))
    # print(np.exp(hmm._log_B))
    # print(np.exp(hmm._log_phi))
