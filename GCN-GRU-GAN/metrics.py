import codecs
import os
import json
from sklearn import metrics

from scipy.stats import wasserstein_distance


def get_wasserstein_distance(seq1, seq2):
    return wasserstein_distance(seq1, seq2)


def get_mean_absolute_error(seq1, seq2):
    metrics.mean_absolute_error(seq1, seq2)
    metrics.r2_score(seq1, seq2)


def get_common_metrics(seq1, seq2):
    return {
        'WSSTD': round(wasserstein_distance(seq1, seq2), 5),
        'RMSE': round(float(metrics.mean_squared_error(seq1, seq2, squared=False)), 5),
        'MAE': round(float(metrics.mean_absolute_error(seq1, seq2)), 5),
        'R^2': round(float(metrics.r2_score(seq1, seq2)), 5),
        'Var': round(float(metrics.explained_variance_score(seq1, seq2)), 5)
    }


def write_metrics(site_path, metrics_, suffix=''):
    with codecs.open(os.path.join(site_path, 'metrics%s.txt' % suffix), "a+", 'utf-8') as text_file:
        text_file.write(json.dumps(metrics_, ensure_ascii=False) + '\n')
