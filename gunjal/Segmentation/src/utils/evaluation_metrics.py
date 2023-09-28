import numpy as np
import sklearn.metrics as skm 

def get_metrics(gt_label, pred_label):
    # accuracy score
    accuracy = skm.accuracy_score(gt_label, pred_label, normalize=True)

    # jaccard score
    jaccard = skm.jaccard_score(gt_label, pred_label, average='micro')

    results = [accuracy, jaccard]
    return results

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_classes):
        mask = (label_true >= 0) & (label_true < n_classes)
        hist = np.bincount(
            n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_classes ** 2,
        ).reshape(n_classes, n_classes)
        return hist
    
    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):   
        hist = self.confusion_matrix
        TP = np.diag(hist)
        TN = hist.sum() - hist.sum(axis = 1) - hist.sum(axis = 0) + np.diag(hist)
        FP = hist.sum(axis = 1) - TP
        FN = hist.sum(axis = 0) - TP
        
        # 1e-6 was added to prevent corner cases where denominator = 0
        
        # Specificity: TN / TN + FP
        specif_cls = (TN) / (TN + FP + 1e-6)
        specif = np.nanmean(specif_cls)
        
        # Senstivity/Recall: TP / TP + FN
        sensti_cls = (TP) / (TP + FN + 1e-6)
        sensti = np.nanmean(sensti_cls)
        
        # Precision: TP / (TP + FP)
        prec_cls = (TP) / (TP + FP + 1e-6)
        prec = np.nanmean(prec_cls)
        
        # F1 = 2 * Precision * Recall / Precision + Recall
        f1 = (2 * prec * sensti) / (prec + sensti + 1e-6)
        
        return (
            {
                "Specificity": specif,
                "Senstivity": sensti,
                "F1": f1,
            }
        )
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        

def val_metric(score):
    # to hold different metrics after each epoch
    Specificity = []
    Senstivity = []
    F1 = []
    acc = []
    js = []

    # Add metrics to empty list above
    Specificity.append(score["Specificity"])
    Senstivity.append(score["Senstivity"])
    F1.append(score["F1"])
    acc.append(score["acc"])
    js.append(score["js"])
