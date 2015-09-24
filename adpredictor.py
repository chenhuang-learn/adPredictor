from scipy.stats import norm
from gaussian_weight import GaussianWeight
import numpy as np
import util
from collections import namedtuple

class AdPredictor(object):
    # get a Config instance: config = Config(beta=1.0, epsilon=1e-3)
    # get attributes: config.beta, config.epsilon
    Config = namedtuple('Config', ['beta', 'epsilon'])

    def __init__(self, config, feature_num = -1):
        # (feature_index -> feature_weight) is stored in self._weights
        self._config = config
        self._feature_num = feature_num
        if feature_num <= 0:
            self._weights = {}
        else:
            self._weights = [GaussianWeight() for i in xrange(feature_num)]

    def predict(self, features):
        # features:[(index,value), (index,value), ...],  output:float
        total_mean, total_variance = self._active_mean_variance(features)
        return norm.cdf(total_mean / total_variance)

    def train(self, features, y, pred=False):
        # y:-1 or 1, features:[(index, value), (index, value), ...]
        total_mean, total_variance = self._active_mean_variance(features)
        v, w = util.gaussian_corrections(y * total_mean / np.sqrt(total_variance))

        for feature_index, feature_value in features:
            weight = self._get_weight(feature_index)
            mean_delta = y * weight._variance / np.sqrt(total_variance) * v * feature_value
            variance_multiplier = 1.0 - weight._variance / total_variance * w * (feature_value ** 2)

            weight.update(mean_delta, variance_multiplier)
            self._apply_dynamics(weight)
            self._set_weight(feature_index, weight)

        if pred:
            return norm.cdf(total_mean / total_variance)

    def save_model(self, file_name):
        handle = open(file_name, 'w')
        handle.write(str(self._feature_num) + ' ' + str(self._config.beta) + '\n')
        if self._feature_num <= 0:
            for feature_index, weight in self._weights.iteritems():
                line = str(feature_index) + ' ' + str(weight._mean) + ' ' + str(weight._variance) + '\n'
                handle.write(line)
        else:
            for feature_index, weight in enumerate(self._weights):
                line = str(feature_index) + ' ' + str(weight._mean) + ' ' + str(weight._variance) + '\n'
                handle.write(line)
        handle.close()

    def load_model(self, file_name):
        if isinstance(self._weights, dict):
            self._weights = {}
        else:
            self._weights = [GaussianWeight() for i in xrange(len(self._weights))]
        line_index = 0
        for line in open(file_name):
            line = line.strip()
            fields = line.split()
            if line_index == 0:
                pass
            else:
                self._weights[int(fields[0])] = GaussianWeight(float(fields[1]), float(fields[2]))
            line_index += 1

    def _active_mean_variance(self, features):
        means = (self._get_weight(f)._mean * v for f, v in features)
        variances = (self._get_weight(f)._variance * (v ** 2) for f, v in features)
        return sum(means), sum(variances) + self._config.beta ** 2

    def _get_weight(self, feature_index):
        if self._feature_num > 0:
            return self._weights[feature_index]
        else:
            return self._weights.get(feature_index, GaussianWeight())

    def _set_weight(self, feature_index, weight):
        self._weights[feature_index] = weight

    def _apply_dynamics(self, weight):
        if self._config.epsilon == 0.0:
            return

        prior = GaussianWeight()
        adjusted_variance = weight._variance * prior._variance / \
            ((1.0 - self._config.epsilon) * prior._variance +
             self._config.epsilon * weight._variance)
        adjusted_mean = adjusted_variance * (
            (1.0 - self._config.epsilon) * weight._mean / weight._variance +
            self._config.epsilon * prior._mean / prior._variance)

        weight._mean = adjusted_mean
        weight._variance = adjusted_variance


