import numpy as np
from sklearn import metrics
from scipy.stats import spearmanr


def compute_metrics(y_true, y_score, task):
    _, task_type = task.task_type
    y_pred = y_score
    if task_type == "multi_class" or task_type == "multi-class":
        y_pred = y_score.argmax(-1)
        scores = task.evaluate(y_true, y_pred)
    elif task_type == "multi_label":
        y_pred = (y_score > 0).astype('float32')
        scores = task.evaluate(y_true, y_pred)
    elif task_type == "binary":
        if isinstance(y_pred, list):
            scores = task.evaluate(y_true, y_pred)
        else:
            y_pred = (y_score > 0).astype('float32')
            scores = task.evaluate(y_true, y_pred)
            scores['accuracy'] = metrics.accuracy_score(y_true, y_pred)
            scores['auc'] = metrics.roc_auc_score(y_true, y_score)
            scores['aupr'] = metrics.average_precision_score(y_true, y_score)

            # import torch
            # scale_size = torch.load('bs_scale_list_ra.npz')
            # roc_auc_list = []
            # auprc_list = []
            # acc_list = []
            # mcc_list = []
            # index = 0
            # for i in scale_size:
            #     y_score_i = y_score[index:index + i].flatten()
            #     y_true_i = y_true[index:index + i].flatten()
            #     y_pred_i = y_pred[index:index + i].flatten()
            #     roc_auc_list.append(metrics.roc_auc_score(y_true_i, y_score_i))
            #     auprc_list.append(metrics.average_precision_score(y_true_i, y_score_i))
            #     acc_list.append(metrics.accuracy_score(y_true_i, y_pred_i))
            #     mcc_list.append(metrics.matthews_corrcoef(y_true_i, y_pred_i))
            #     index += i
            # torch.save(roc_auc_list, 'bs_model_our_roc_auc_list_ra')
            # torch.save(auprc_list, 'bs_model_our_auprc_list_ra')
            # torch.save(acc_list, 'bs_model_our_acc_list_ra')
            # torch.save(mcc_list, 'bs_model_our_mcc_list_ra')
            # torch.save(y_pred, 'bs_model_64_y_pred_seq.npz')
            # torch.save(y_true, 'bs_model_64_y_true_seq.npz')
            # torch.save(y_score, 'bs_model_64_y_score_seq.npz')

    elif task_type == 'regression':
        scores = task.evaluate(y_true, y_pred)
        scores['neg_mse'] = -scores['mse']
        scores['mae'] = metrics.mean_absolute_error(y_true, y_score)
        scores['spearmanr'] = spearmanr(y_true, y_pred).correlation
        scores['r2'] = metrics.r2_score(y_true, y_score)
    else:
        scores = task.evaluate(y_true, y_pred)
    return scores
