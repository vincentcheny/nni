# BSD 3-Clause License
# Copyright (c) 2017-2018, ML4AAD
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
from argparse import Namespace
import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
np.random.seed(0)
import scipy.stats as sps
import statsmodels.api as sm

import pickle
from dragonfly.utils.option_handler import get_option_specs, load_options
from dragonfly import load_config, multiobjective_maximise_functions

logger = logging.getLogger('BOHB_Advisor')


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


class CG_BOHB:
    def __init__(self, configspace, min_points_in_model=None,
                 top_n_percent=15, num_samples=64, random_fraction=1/3,
                 bandwidth_factor=3, min_bandwidth=1e-3):
        """Fits for each given budget a kernel density estimator on the best N percent of the
        evaluated configurations on this budget.


        Parameters:
        -----------
        configspace: ConfigSpace
            Configuration space object
        top_n_percent: int
            Determines the percentile of configurations that will be used as training data
            for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
            for training.
        min_points_in_model: int
            minimum number of datapoints needed to fit a model
        num_samples: int
            number of samples drawn to optimize EI via sampling
        random_fraction: float
            fraction of random configurations returned
        bandwidth_factor: float
            widens the bandwidth for contiuous parameters for proposed points to optimize EI
        min_bandwidth: float
            to keep diversity, even when all (good) samples have the same value for one of the parameters,
            a minimum bandwidth (Default: 1e-3) is used instead of zero.
        """
        self.top_n_percent = top_n_percent
        self.configspace = configspace
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters())+1

        if self.min_points_in_model < len(self.configspace.get_hyperparameters())+1:
            logger.warning('Invalid min_points_in_model value. Setting it to %i', len(self.configspace.get_hyperparameters()) + 1)
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        hps = self.configspace.get_hyperparameters()


        self.kde_vartypes = ""
        self.vartypes = []

        for h in hps:
            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)

        # store precomputed probs for the categorical parameters
        self.cat_probs = []

        self.configs = dict()
        self.losses = dict()
        self.good_config_rankings = dict()
        self.kde_models = dict()
        self.runtime = dict()
        self.search_space = dict()
        self.is_moo = False

    def largest_budget_with_model(self):
        if not self.kde_models:
            return -float('inf')
        return max(self.kde_models.keys())

    def sample_from_largest_budget(self, info_dict):
        """We opted for a single multidimensional KDE compared to the
        hierarchy of one-dimensional KDEs used in TPE. The dimensional is
        seperated by budget. This function sample a configuration from
        largest budget. Firstly we sample "num_samples" configurations,
        then prefer one with the largest l(x)/g(x).

        Parameters:
        -----------
        info_dict: dict
            record the information of this configuration

        Returns
        -------
        dict:
            new configuration named sample
        dict:
            info_dict, record the information of this configuration
        """
        best = np.inf
        best_vector = None

        # print(f"[vincent] self.kde_models.keys():{self.kde_models.keys()}")
        budget = max(self.kde_models.keys())

        # print(f"[vincent] self.kde_models[budget]['good']:{self.kde_models[budget]['good']}")
        l = self.kde_models[budget]['good'].pdf # possibility density function
        g = self.kde_models[budget]['bad'].pdf

        minimize_me = lambda x: max(1e-32, g(x))/max(l(x), 1e-32)

        kde_good = self.kde_models[budget]['good']
        kde_bad = self.kde_models[budget]['bad']

        for i in range(self.num_samples):
            idx = np.random.randint(0, len(kde_good.data))
            datum = kde_good.data[idx]
            vector = []

            for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                bw = max(bw, self.min_bandwidth)
                if t == 0:
                    bw = self.bw_factor*bw
                    vector.append(sps.truncnorm.rvs(-m/bw, (1-m)/bw, loc=m, scale=bw))
                else:
                    if np.random.rand() < (1-bw):
                        vector.append(int(m))
                    else:
                        vector.append(np.random.randint(t))
            val = minimize_me(vector)

            if not np.isfinite(val):
                logger.warning('sampled vector: %s has EI value %s', vector, val)
                logger.warning("data in the KDEs:\n%s\n%s", kde_good.data, kde_bad.data)
                logger.warning("bandwidth of the KDEs:\n%s\n%s", kde_good.bw, kde_bad.bw)
                logger.warning("l(x) = %s", l(vector))
                logger.warning("g(x) = %s", g(vector))

                # right now, this happens because a KDE does not contain all values for a categorical parameter
                # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde,
                # so it shouldn't be terrible.
                if np.isfinite(l(vector)):
                    best_vector = vector
                    break

            if val < best:
                best = val
                best_vector = vector

        if best_vector is None:
            logger.debug("Sampling based optimization with %i samples failed -> using random configuration", self.num_samples)
            sample = self.configspace.sample_configuration().get_dictionary()
            info_dict['model_based_pick'] = False

        else:
            logger.debug('best_vector: %s, %s, %s, %s', best_vector, best, l(best_vector), g(best_vector))
            for i, _ in enumerate(best_vector):
                hp = self.configspace.get_hyperparameter(self.configspace.get_hyperparameter_by_idx(i))
                if isinstance(hp, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                    best_vector[i] = int(np.rint(best_vector[i]))
            sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()

            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample)
            info_dict['model_based_pick'] = True

        return sample, info_dict

    def get_config_old(self, budget):
        """Function to sample a new configuration
        This function is called inside BOHB to query a new configuration

        Parameters:
        -----------
        budget: float
            the budget for which this configuration is scheduled

        Returns
        -------
        config
            return a valid configuration with parameters and budget
        """
        logger.debug('start sampling a new configuration.')
        sample = None
        info_dict = {}

        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if not self.kde_models.keys() or np.random.rand() < self.random_fraction:
            logger.debug(np.random.rand())
            logger.debug(type(self.configspace.get_hyperparameters()))
            logger.debug(type(self.configspace.get_hyperparameter("BATCH_SIZE")))
            logger.debug(type(self.configspace.get_hyperparameter("EPSILON")))
            # logger.debug(self.configspace.get_hyperparameter("EPSILON")._sample(np.random.RandomState(0),1))
            # logger.debug(self.configspace.get_hyperparameter("EPSILON").get_choices())
            # logger.debug(self.configspace.get_hyperparameter("EPSILON").choices)

            sample = self.configspace.sample_configuration()
            info_dict['model_based_pick'] = False

        
        if sample is None:
            sample, info_dict = self.sample_from_largest_budget(info_dict)
        print(f'[vincent] sample before deactivate: {sample.get_dictionary()}')
        '''
        Configuration:
            batch_size, Value: 128
            dense_size, Value: 1024
            epoch, Value: 10
            filter_num, Value: 48
            kernel_size, Value: 5
            learning_rate, Value: 0.05
            optimizer, Value: 'sgd'
            weight_decay, Value: 0.0001
        '''
        sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
            configuration_space=self.configspace,
            configuration=sample.get_dictionary()
        ).get_dictionary()
        print(f'[vincent] sample after deactivate: {sample}')
        '''
        {'batch_size': 128, 'dense_size': 1024, 'epoch': 10, 'filter_num': 48, 'kernel_size': 5, 'learning_rate': 0.05, 'optimizer': 'sgd', 'weight_decay': 0.0001}
        '''
        logger.debug('done sampling a new configuration.')
        sample['TRIAL_BUDGET'] = budget

        # logger.debug(f'[vincent] sample from get_config_old:{sample}')
        return sample
    
    def get_config(self, budget):
        """Function to sample a new configuration
        This function is called inside BOHB to query a new configuration

        Parameters:
        -----------
        budget: float
            the budget for which this configuration is scheduled

        Returns
        -------
        config
            return a valid configuration with parameters and budget
        """
        if not self.is_moo:
            return self.get_config_old(budget)

        logger.debug('start sampling a new configuration.')
        if not self.configs:
            print(f"[vincent] self.configs is empty! Use a random config instead.")
            sample = self.configspace.sample_configuration()
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary()
            ).get_dictionary()
            sample['TRIAL_BUDGET'] = budget
            return sample
        domain_vars = list()
        for name in self.search_space.keys():
            if isinstance(self.search_space[name][0], (float, int)):
                var_type = 'discrete_numeric'
            else:
                var_type = 'discrete'
            domain_var = {'type': var_type, 'items':self.search_space[name]}
            domain_vars.append(domain_var)
        points = list()
        vals = list()
        true_vals = list()
        
        print(f"[vincent] self.configs:{self.configs} budget:{budget}")
        print(f"{list(self.search_space.keys())}")
        for conf_array in self.configs[0]:
            first, second = [], []
            for i in range(len(conf_array)):
                item = self.search_space[list(self.search_space.keys())[i]][int(conf_array[i])]
                if isinstance(item, (float,int)):
                    second.append(item)
                else:
                    first.append(item)
            points.append([first,second])
        for idx in range(len(self.losses[0])):
            vals.append([-self.losses[0][idx], -self.runtime[0][idx]])
            true_vals.append([-self.losses[0][idx], -self.runtime[0][idx]])
        
        print(f"[vincent] len of points:{len(points)}")
        if len(points) > 10:
            vals_array = np.array(vals)
            pareto_index = is_pareto_efficient_simple(vals_array)
            p_idx = []
            np_idx = []
            np_items = []
            for j in range(len(pareto_index)):
                if pareto_index[j] == True:
                    p_idx.append(j)
                else:
                    np_idx.append(j)
                    np_items.append(vals[j])
            print(f"[vincent] pareto_index:{p_idx}")
            print(f"[vincent] not pareto_index:{np_idx}")


            if len(p_idx) >= 5:
                tmp_idx = []
                for j in range(5):
                    tmp_idx.append(p_idx[j])
                points = [points[i] for i in tmp_idx]
                vals = [vals[i] for i in tmp_idx]
                true_vals = [true_vals[i] for i in tmp_idx]
            else:
                num_diff = 5 - len(p_idx)
                print(f"[vincent] diff num:{num_diff}")
                print(f"[vincent] search space:{self.search_space}")
                if self.search_space['PREFERENCE'][0] == "accuracy":
                    acc_items = [-item[0] for item in np_items]
                    sort_n_idx = np.argsort(acc_items)
                    for i in range(num_diff):
                        p_idx.append(sort_n_idx[i])
                    print(f"[vincent] final pareto_index:{p_idx}")
                    points = [points[i] for i in p_idx]
                    vals = [vals[i] for i in p_idx]
                    true_vals = [true_vals[i] for i in p_idx]
                elif self.search_space['PREFERENCE'][0] == "runtime":
                    time_items = [-item[1] for item in np_items]
                    sort_n_idx = np.argsort(time_items)
                    for i in range(num_diff):
                        p_idx.append(sort_n_idx[i])
                    print(f"[vincent] final pareto_index:{p_idx}")
                    points = [points[i] for i in p_idx]
                    vals = [vals[i] for i in p_idx]
                    true_vals = [true_vals[i] for i in p_idx]

            # import random
            # idx_list = random.sample(range(len(points)), 10)
            # print(f"[vincent] random selections list idx_list:{idx_list}")
            # points = [points[i] for i in idx_list]
            # vals = [vals[i] for i in idx_list]
            # true_vals = [true_vals[i] for i in idx_list]

        ## vals = [[acc,-spent time],[acc,-spent time]]
        ## load from memory
        previous_eval = {'qinfos':[]}
        for i in range(len(points)):
            tmp = Namespace(point=points[i],val=vals[i],true_val=true_vals[i])
            previous_eval['qinfos'].append(tmp)
        p = Namespace(**previous_eval)
        load_args = [
                get_option_specs('init_capital', False, 1, 'Path to the json or pb config file. '),
                get_option_specs('init_capital_frac', False, None,'The fraction of the total capital to be used for initialisation.'),
                get_option_specs('num_init_evals', False, 1,'The number of evaluations for initialisation. If <0, will use default.'),
                get_option_specs('prev_evaluations', False, p,'Data for any previous evaluations.')
        ]
        options = load_options(load_args)
        config_params = {'domain': domain_vars}
        config = load_config(config_params)
        max_num_evals = 1
        self.dragonfly_config = None
        
        def fake_func(x):
            if not self.dragonfly_config:
                self.dragonfly_config = x
                print(f"[vincent] x is assigned to self.dragonfly_config:{self.dragonfly_config}")
            return 0

        moo_objectives = [fake_func, fake_func]
        _, _, _ = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='num_evals',config=config,options=options)
        print(f"[vincent] self.dragonfly_config after dragonfly:{self.dragonfly_config}")


        ## load prev from the file
        # data_to_save = {'points': points,
        #                 'vals': vals,
        #                 'true_vals': true_vals}
        # print(f"[vincent] data_to_save:{data_to_save}")
        # temp_save_path = './dragonfly.saved'
        
        # with open(temp_save_path, 'wb') as save_file_handle:
        #     pickle.dump(data_to_save, save_file_handle)

        
        # load_args = [
        #     get_option_specs('progress_load_from', False, temp_save_path,
        #     'Load progress (from possibly a previous run) from this file.') 
        # ]
        # options = load_options(load_args)
        # config_params = {'domain': domain_vars}
        # config = load_config(config_params)
        # max_num_evals = 1
        # self.dragonfly_config = None

        # def fake_func(x):
        #     if not self.dragonfly_config:
        #         self.dragonfly_config = x
        #         print(f"[vincent] x is assigned to self.dragonfly_config:{self.dragonfly_config}")
        #     return 0
        
        # moo_objectives = [fake_func, fake_func]
        # _, _, _ = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='num_evals',config=config,options=options)
        # print(f"[vincent] self.dragonfly_config after dragonfly:{self.dragonfly_config}")
        # import os
        # if os.path.exists(temp_save_path):
        #     os.remove(temp_save_path)

        if not self.dragonfly_config:
            print(f"[vincent] Get empty config from dragonfly! Use a random config instead.")
            sample = self.configspace.sample_configuration()
        else:
            sample = dict()
            df_idx = 0
            for name in self.search_space.keys():
                sample[name] = self.dragonfly_config[df_idx]
                df_idx += 1

        logger.debug('done sampling a new configuration.')
        sample['TRIAL_BUDGET'] = budget

        print(f'[vincent] sample from get_config:{sample}')

        return sample

    def impute_conditional_data(self, array):
        return_array = np.zeros(array.shape)
        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()
            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()
                if valid_indices:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]
                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)
                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array

    def new_result_old(self, loss, budget, parameters, update_model=True):
        """
        Function to register finished runs. Every time a run has finished, this function should be called
        to register it with the loss.

        Parameters:
        -----------
        loss: float
            the loss of the parameters
        budget: float
            the budget of the parameters
        parameters: dict
            the parameters of this trial
        update_model: bool
            whether use this parameter to update BP model

        Returns
        -------
        None
        """
        if loss is None:
            # One could skip crashed results, but we decided
            # assign a +inf loss and count them as bad configurations
            loss = np.inf

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []
            self.runtime[budget] = []

        # skip model building if we already have a bigger model
        if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
            return

        # We want to get a numerical representation of the configuration in the original space
        conf = ConfigSpace.Configuration(self.configspace, parameters)
        # print(f"[vincent] conf in new_result_old:{conf}")
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(loss)
        # self.runtime[budget].append(runtime)

        # skip model building:
        # a) if not enough points are available
        if len(self.configs[budget]) <= self.min_points_in_model - 1:
            logger.debug("Only %i run(s) for budget %f available, need more than %s \
            -> can't build model!", len(self.configs[budget]), budget, self.min_points_in_model+1)
            return
        # b) during warnm starting when we feed previous results in and only update once
        if not update_model:
            return

        train_configs = np.array(self.configs[budget])
        train_losses = np.array(self.losses[budget])
        # train_runtime = np.array(self.runtime[budget])

        
        n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0])//100)
        n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100)

        # Refit KDE for the current budget
        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        #more expensive crossvalidation method
        #bw_estimation = 'cv_ls'
        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes, bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes, bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models[budget] = {
            'good': good_kde,
            'bad' : bad_kde
        }
        logger.debug(f"[vincent] self.kde_models.keys():{self.kde_models.keys()} self.kde_models[budget]:{self.kde_models[budget]}")
        # update probs for the categorical parameters for later sampling
        logger.debug('done building a new model for budget %f based on %i/%i split\nBest loss for this budget:%f\n',
                    budget, n_good, n_bad, np.min(train_losses))
    

    def new_result(self, data, budget, parameters, update_model=True):
        """
        Function to register finished runs. Every time a run has finished, this function should be called
        to register it with the loss.

        Parameters:
        -----------
        data: float
            [accuracy, runtime]
        budget: float
            the budget of the parameters
        parameters: dict
            the parameters of this trial
        update_model: bool
            whether use this parameter to update BP model

        Returns
        -------
        None
        """
        self.is_moo = True
        if not self.search_space:
            for hp in self.configspace.get_hyperparameters():
                self.search_space[hp.name] = list(hp.choices)

        if data is None:
            # One could skip crashed results, but we decided
            # assign a +inf loss and count them as bad configurations
            data = {'accuracy':0, 'runtime':np.inf}

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []
            self.runtime[budget] = []

        # We want to get a numerical representation of the configuration in the original space
        conf = ConfigSpace.Configuration(self.configspace, parameters)
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(data['accuracy']) # negative
        self.runtime[budget].append(data['runtime']) # positive
