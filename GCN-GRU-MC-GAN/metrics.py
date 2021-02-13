import codecs
import os
import json

from scipy.stats import wasserstein_distance


def get_wasserstein_distance(seq1, seq2):
    return wasserstein_distance(seq1, seq2)


def write_metrics(metrics):
    with codecs.open(os.path.join('generated/week_dist/metrics.txt'), "a+", 'utf-8') as text_file:
        text_file.write(json.dumps(metrics, ensure_ascii=False) + '\n')
