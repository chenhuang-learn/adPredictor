import sys
from scipy.stats import norm
from util import kl_divergence

# sys.argv[1]: input_model_file
# sys.argv[2]: average_feature_num_per_sample - 1
# sys.argv[3]: output_feature_importance_file

# assume every feature_value is 1

importances = []
extra_feature_num = int(sys.argv[2])

line_index = 0
for line in open(sys.argv[1]):
    line = line.strip()
    fields = line.split()
    if line_index == 0:
        beta = float(fields[1])
    else:
        feature_index = int(fields[0])
        mean = float(fields[1])
        variance = float(fields[2])
        prob = norm.cdf(mean / (variance + extra_feature_num + beta ** 2))
        impor = kl_divergence(prob, 0.5)
        importances.append((feature_index, impor))
    line_index += 1

importances_list = sorted(importances, key=lambda x:x[1], reverse=True)
output_handle = open(sys.argv[3], 'w')
for fea_index, impor in importances_list:
    line = str(fea_index) + " " + str(impor) + "\n"
    output_handle.write(line)
output_handle.close()

