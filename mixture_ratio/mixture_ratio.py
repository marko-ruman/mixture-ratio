from scipy import special as sfunc
from scipy.optimize import fsolve
import numpy as np
from tqdm import tqdm


def update_statistics_parallel(statistic_L_i):
    """Parallel updating of the sufficient statistics with the current data.

    Parameters
    ----------
    statistic_L_i : tuple

        Tuple containing previous statistics, factor vector "L" and index of the component "i"
    """
    statistics = statistic_L_i[0]
    L = statistic_L_i[1]
    i = statistic_L_i[2]
    return i, fsolve(discrete_optimised_function, statistics, args=(L, ))


def discrete_optimised_function(stat_vec, L_component):
    """Compute value of the optimised function for the given statistics vector and the L vector for
        the specific mixture component

    Parameters
    ----------
    stat_vec : array-like, shape (n_params of the component)
        Sufficient statistics for the specific component

    L_component: array-like shape (n_params of the component)
        Factors needed for the projection for the specific component

    Returns
    -------
    array, shape(stat_vec.shape[0], )
    """
    f = sfunc.digamma(stat_vec)-sfunc.digamma(np.sum(stat_vec)) - L_component
    return f


class MixtureRatio:
    """
        Mixture Ratio model

        Parameters
        ----------
        variables_domain : array-like, shape (X.shape[1]+1,)
            Domain of all variables, i.e. number of possible values.

        variables_connection : list of array-likes with shape (n_variables in i-th component,  )
            Defining of the structure of the Mixture Ratio. If none, then standard connection
            [[0, 1], [0, 2], ... [0, X.shape[1]]

        init_statistics : list of arrays, shape (n_components + 1, n_parameters for each component)
            Sufficient statistics defining the prior distribution on parameters of the Mixture Ratio model

        pool : Pool() from the multiprocessing package
            Pool used for parallel updating of the Mixture components. If None, the components are updated sequentially.

        Attributes
        ----------
        statistics : list of arrays, shape (n_components + 1, n_parameters for each component)
            Sufficient statistics defining the learnt distribution on parameters of the Mixture Ratio model

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 0], [1, 1], [1, 1], [2, 1], [3, 2]])
        >>> Y = np.array([0, 0, 0, 1, 1, 1])
        >>> from mixture_ratio import MixtureRatio
        >>> mix = MixtureRatio(variables_domain=[2, 4, 3])
        >>> mix.fit(X, Y)
        >>> print(mix.predict_proba([[0, 1]]))
        [1]
        """
    def __init__(self, variables_domain, variables_connection=None, init_statistics=None, pool=None):
        self.variables_domain = np.array(variables_domain).astype(int)
        if variables_connection is None:
            self.variables_connection = self._get_standard_variables_connection()
        else:
            self.variables_connection = variables_connection
        self.number_of_components = len(self.variables_connection)
        self.current_data = []
        self.normalizing_constants = self._get_normalizing_constants()

        if init_statistics is None:
            self.statistics = self._get_uniform_statistics()
        else:
            self.statistics = init_statistics
        self.expected_parameters = []
        self._update_expected_parameters()
        self.pool = pool

    @staticmethod
    def normalize_proba(P):
        """Normalize conditional probability P, so that it sums to one for each condition.

        Parameters
        ----------
        P : array-like
            Probability to be normalized.

        Returns
        -------
        P : array-like
            Normalized probability.
        """
        row_sums = P.sum(axis=1)
        P = P / row_sums[:, np.newaxis]
        return P

    @staticmethod
    def k_delta(x, y):
        """Kronecked delta function of x and y

        Parameters
        ----------
        x : number

        y : number

        Returns
        -------
        f : number
        Returns 1 if x=y, 0 otherwise
        """
        f = 0
        if x == y:
            f = 1
        return f

    @staticmethod
    def k_delta_vec(dimension, position):
        """Create an array with 1 on the given position, 0 on other positions.

        Parameters
        ----------
        dimension : integer
            Dimension of the created array.

        position : integer
            Position of 1 in the created array

        Returns
        -------
        delta_vec : array, shape (dimension, )
            Array with 1 on the given position, 0 on other positions
        """
        delta_vec = np.zeros(dimension)
        delta_vec[position] = 1
        return delta_vec

    def _get_standard_variables_connection(self):
        """Create a default mixture structure with component connections [[0, 1], [0, 2], ..., [0, n_features]]

        Returns
        -------
        standard_connection : list of arrays with shape (2, )
            List of arrays with default variable connections.
        """
        standard_connection = []
        for i in range(len(self.variables_domain)-1):
            standard_connection.append(np.array([0, i+1]).astype(int))
        return standard_connection

    def _get_uniform_statistics(self):
        """Create a default mixture structure with component connections [[0, 1], [0, 2], ..., [0, n_features]]

        Returns
        -------
        uniform_statistics : list of arrays with shape (product of variables domain in particular connection, )
            List of arrays with initial statistics defining uniform parameter distribution.
        """
        uniform_statistics = []
        uniform_statistics.append(np.ones(shape=self.number_of_components,))
        for i in range(self.number_of_components):
            num_of_values = np.prod(self.variables_domain[self.variables_connection[i]])
            uniform_statistics.append(np.ones(shape=num_of_values,))
        return uniform_statistics

    def _get_normalizing_constants(self):
        """Compute the normalizing constants for each connection depending on the mixture structure.

         Returns
        -------
        normalizing_constants: list of arrays with shape (product of variables domain in particular connection, )
            List of arrays with normalizing constants.
        """
        normalizing_constants = []
        for i in range(self.number_of_components):
            normalizing_constants.append(1 / np.prod(self.variables_domain[self.variables_connection[i]]))
        return normalizing_constants

    def fit(self, X, y):
        """Fit Mixture Ratio according to X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks (even on single observations) of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once or when the task requires online learning.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        data = np.zeros((X.shape[0], X.shape[1] + 1))
        data[:, 0] = y
        data[:, 1:] = X
        data = data.astype(int)
        for i in tqdm(range(data.shape[0])):
            self._update_current_data(data[i, :])
            self._update_statistics()
        return self

    def _update_current_data(self, data):
        """Update attribute current_data with the newest data array

        Parameters
        ----------
        data : array-like, shape (X.shape[0], X.shape[1]+1)
            One observation vector of y and X
        """
        self.current_data = data

    def _update_statistics(self):
        """Update the sufficient statistics with the current data - self.current_data

        """
        if len(self.variables_connection) > 1:
            # Updating for a mixture with at least 2 components
            L = self._get_L()
            if self.pool is not None:
                # If multiprocessing pool was provided, the update is parallelized
                statistics_tuples = [(self.statistics[i], L[i], i) for i in range(len(self.statistics))]
                results = self.pool.map(update_statistics_parallel, statistics_tuples)
                for result in results:
                    self.statistics[result[0]] = result[1]
            else:
                # Non-paralelized update of statistics
                for i in range(len(self.statistics)):
                    self.statistics[i] = fsolve(self._discrete_optimised_function, self.statistics[i], args=(L[i]))
        else:
            # Updating for a mixture with only 1 component
            self._update_statistics_full_table()
        # Updating of the expected values of parameters
        self._update_expected_parameters()

    def _discrete_optimised_function(self, stat_vec, L_component):
        """Compute value of the optimised function for the given statistics vector and the L vector for
        the specific miture component

        Parameters
        ----------
        stat_vec : array-like, shape (n_params of the component)
            Sufficient statistics for the specific component

        L_component: array-like shape (n_params of the component)
            Factors needed for the projection for the specific component

        Returns
        -------
        array, shape(stat_vec.shape[0], )
        """

        return sfunc.digamma(stat_vec) - sfunc.digamma(np.sum(stat_vec)) - L_component

    def _update_statistics_full_table(self):
        """Update the sufficient statistics when the mixture has only one component.
        """
        ind = np.ravel_multi_index(self.current_data[self.variables_connection[0]],
                                   self.variables_domain[self.variables_connection[0]])
        self.statistics[1][ind] += 1

    def _get_L1(self, gamma, H):
        """Compute the approximated normalizing factor used in Bayes rule for the current learning step.

        Parameters
        ----------
        gamma : array-like, shape (n_components,)
            A part of the normalizing factor specific for each component.

        H: array-like, shape (n_component, )
            The denominator of the Mixture Ratio computed for the expected parameters specific for each component.

        Returns
        ----
        L1: number
        The approximated normalizing factor for Bayes rule---

        """
        L1 = np.sum([g * H for g, H in zip(gamma, H)])
        return L1

    def _get_L(self):
        """Compute the approximated normalizing factor for Bayes rule.

        Returns
        -------
        L: list of arrays with shapes (product of variables domain in particular connection, )
        The approximated factors needed for the Kullback-Leibler projections
        """
        parameters = self._get_next_expected_parameters()
        H, grad_H = self._get_H_with_grad(parameters)
        gamma = self._get_gamma()
        L1 = self._get_L1(gamma, H)

        L = []
        L.append(np.zeros([self.number_of_components]))
        stat_sum_0 = np.sum(self.statistics[0])+1

        for c in range(self.number_of_components):

            L[0][c] = np.sum(np.multiply(gamma,
                                         np.multiply(H, sfunc.digamma(self.statistics[0][c]
                                                                      + self.k_delta_vec(self.number_of_components, c))
                                                     - sfunc.digamma(stat_sum_0))
                                         + 1/stat_sum_0*np.sum(
                                             np.multiply(parameters[0][:, :].transpose(),
                                                         (grad_H[c, :].reshape(-1, 1) - grad_H.transpose())), 1)))

            ind = np.ravel_multi_index(self.current_data[self.variables_connection[c]],
                                       self.variables_domain[self.variables_connection[c]])
            stat_sum = np.sum(self.statistics[c + 1])

            L.append(np.zeros(np.prod(self.variables_domain[self.variables_connection[c]])))

            for d in range(np.prod(self.variables_domain[self.variables_connection[c]])):

                L[c + 1][d] = np.sum(
                                np.multiply(
                                    gamma, np.multiply(H,
                                                       sfunc.digamma(self.statistics[c+1][d]
                                                                     + self.k_delta_vec(self.number_of_components, c)
                                                                     *self.k_delta(d, ind))
                                                       - sfunc.digamma(stat_sum + self.k_delta_vec(
                                                           self.number_of_components, c)))))
        L = L/L1
        return L

    def _update_expected_parameters(self):
        """Update expected values of all parameters. Typically called after the update of statistics.
        """
        self.expected_parameters = []
        for i in range(self.number_of_components+1):
            self.expected_parameters.append(np.true_divide(self.statistics[i], np.sum(self.statistics[i])))

    def _get_next_expected_parameters(self):
        """Compute the expected parameters needed for the projection of the distribution obtained by Bayes rule.

         Returns
        -------
        next_expected_parameters: list of arrays with shapes (product of variables domain in particular connection, )
        The expected values of parameters specific for each mixture component according to the distribution obtained
        by Bayes rule
        """
        next_expected_parameters = []

        statistic_sum = np.sum(self.statistics[0])
        param = np.array([statistic_sum / (statistic_sum + 1)*self.expected_parameters[0][:]
                          + 1/(statistic_sum+1)*self.k_delta_vec(self.number_of_components, c)
                          for c in range(self.number_of_components)]).transpose()
        next_expected_parameters.append(param)
        for i in range(self.number_of_components):
            statistic_sum = np.sum(self.statistics[i+1])
            ind = np.ravel_multi_index(self.current_data[self.variables_connection[i]],
                                       self.variables_domain[self.variables_connection[i]])

            param = np.array([self.expected_parameters[i+1][:]
                              for _ in range(self.number_of_components)]).transpose()

            param[:, i] = statistic_sum / (statistic_sum + 1)*param[:, i] + 1/(statistic_sum+1) \
                          * self.k_delta_vec(np.prod(self.variables_domain[self.variables_connection[i]]), ind)

            next_expected_parameters.append(param)
        return next_expected_parameters

    def _get_H_with_grad(self, params):
        """Compute the denominator and its gradient of the Mixture Ratio for the given parameters.

        Returns
        -------
        H: array, shape (n_components, )

        grad_H: array, shape(n_components, n_components)
        """
        H = np.zeros(self.number_of_components)
        grad_H = np.zeros((self.number_of_components, self.number_of_components))
        for c in range(self.number_of_components):
            for d in range(self.number_of_components):
                for i in range(self.variables_domain[0]):
                    data = np.copy(self.current_data)
                    data[0] = i
                    ind = np.ravel_multi_index(data[self.variables_connection[d]],
                                               self.variables_domain[self.variables_connection[d]])
                    H[c] = H[c] + params[0][d, c] * params[d+1][ind, c] / self.normalizing_constants[d]
                    grad_H[d, c] = grad_H[d, c] + params[d+1][ind, c]/self.normalizing_constants[d]
            H[c] = 1/H[c]
            grad_H[:, c] = -np.power(H[c], 2)*grad_H[:, c]

        return (H, grad_H)

    def _get_gamma(self):
        """Compute the part of the normalizing factor, gamma,  specific for each component.

        Returns
        -------
        gamma: array, shape (n_components, )
        """
        gamma = []
        for i in range(self.number_of_components):
            ind = np.ravel_multi_index(self.current_data[self.variables_connection[i]],
                                       self.variables_domain[self.variables_connection[i]])
            gamma.append(self.expected_parameters[0][i]
                         * self.expected_parameters[i+1][ind] / self.normalizing_constants[i])
        return gamma

    def get_predictor(self):
        """Compute the probability for the whole variables domain.

        Returns
        -------
        P: array, shape (tuple(self.variables_domain))
        Array representing the conditional probability function.
        """
        P = np.zeros(tuple(self.variables_domain, ))
        for i in range(np.prod(self.variables_domain)):
            data = np.array(np.unravel_index(i, self.variables_domain))
            for c in range(self.number_of_components):
                ind = np.ravel_multi_index(data[self.variables_connection[c]],
                                           self.variables_domain[self.variables_connection[c]])

                P[tuple(data)] = P[tuple(data)] + self.expected_parameters[0][c]\
                                 * self.expected_parameters[c+1][ind]/self.normalizing_constants[c]
        P = self.normalize_proba(P)
        return P

    def predict_proba(self, X):
        """Compute the predicted probability for all of the classes for the given X.

       Returns
       -------
       proba: array, shape (n_classes, )
       Array containing probabilities of each class.
       """
        proba = np.zeros((len(X), self.variables_domain[0]))
        for i, d in enumerate(X):
            P = np.zeros(self.variables_domain[0])
            for o in range(self.variables_domain[0]):
                data_vec = np.concatenate((np.array([o]), d))
                for c in range(self.number_of_components):
                    ind = np.ravel_multi_index(data_vec[self.variables_connection[c]],
                                               self.variables_domain[self.variables_connection[c]])
                    P[o] = P[o] + self.expected_parameters[0][c]\
                           * self.expected_parameters[c + 1][ind] / self.normalizing_constants[c]
            proba[i, :] = P[:]/np.sum(P)
        return proba

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)