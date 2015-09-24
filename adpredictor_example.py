from adpredictor import AdPredictor
from sampler import FileSampler
from sklearn.metrics import roc_auc_score
from collections import namedtuple
from multiprocessing import Pool

Config = namedtuple('Config',
        ['epsilon', 'beta', 'feature_num', 'train_files', 'test_files', 'model_file'])

def libsvm_file_process(config):
    adpredictor = AdPredictor(AdPredictor.Config(config.beta, config.epsilon), config.feature_num)
    train_pred, train_label = [], []
    for train_file in config.train_files:
        for features, label in FileSampler(train_file).generate_samples():
            prob = adpredictor.train(features, label, True)
            train_pred.append(prob)
            train_label.append(1 if label > 0 else 0)
    adpredictor.save_model(config.model_file)
    test_pred, test_label = [], []
    for test_file in config.test_files:
        for features, label in FileSampler(test_file).generate_samples():
            prob = adpredictor.predict(features)
            test_pred.append(prob)
            test_label.append(1 if label>0 else 0)
    return train_pred, train_label, test_pred, test_label, config

epsilons = [0.0]
betas = [0.5, 1.0, 5.0]
feature_num = 123

train_files = ["a9a_train.pls"]
test_files = ["a9a_test.pls"]
model_folder = "models/"

configs = []
for epsilon in epsilons:
    for beta in betas:
        model_file = model_folder + 'model_' + str(beta) + '_' + str(epsilon)
        config = Config(epsilon, beta, feature_num, train_files, test_files, model_file)
        configs.append(config)

# p = Pool(4)
# results = p.map(libsvm_file_process, configs)
results = []
for config in configs:
    results.append(libsvm_file_process(config))

for result in results:
    print str(result[4].beta) + "_" + str(result[4].epsilon) + " " + \
            str(roc_auc_score(result[1], result[0])) + " " + str(roc_auc_score(result[3], result[2]))

