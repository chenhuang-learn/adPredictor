
class GaussianWeight(object):
    def __init__(self, mean=0.0, variance=1.0):
        self._mean = mean
        self._variance = variance

    def update(self, mean_delta, variance_multiplier):
        self._mean += mean_delta
        self._variance *= variance_multiplier

    def __repr__(self):
        return str(self._mean) + " " + str(self._variance)

if __name__ == "__main__":
    weight = GaussianWeight()
    weight.update(0.1, 0.8)
    print weight

