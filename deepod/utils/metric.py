import numpy as np
from sklearn import metrics


def _get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def cal_metrics(label, score, pa=False):
    if pa:
        score = _adjust_scores(label, score)
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    best_f1, best_p, best_r = _get_best_f1(label, score)

    return auroc, ap, best_f1, best_p, best_r


def _adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]

    Parameters
    ----------
    label: np.array, required
        data label, 0 indicates normal timestamp, and 1 is anomaly

    score: np.array, required
        anomaly score, higher score indicates higher likelihoods to be anomaly
    """
    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score

def evaluate(y_true, scores):
    """calculate evaluation metrics"""
    roc_auc = metrics.roc_auc_score(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)

    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true==0)[0]) / len(y_true)
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = y_true.astype(int)
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f_score