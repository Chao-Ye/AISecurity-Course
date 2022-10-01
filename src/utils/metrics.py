from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def cal_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')

def cal_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def cal_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

