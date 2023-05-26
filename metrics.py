def get_set_precision(y_true, y_pred):
    precision = len(set(y_true).intersection(set(y_pred))) / (
        1 if len(y_pred) == 0 else len(y_pred)
    )
    return precision * 100.0


def get_set_recall(y_true, y_pred):
    recall = len(set(y_true).intersection(set(y_pred))) / (
        1 if len(y_true) == 0 else len(y_true)
    )
    return recall * 100.0


def get_set_f1(y_true, y_pred):
    precision = get_set_precision(y_true, y_pred) / 100.0
    recall = get_set_recall(y_true, y_pred) / 100.0
    f1 = (2 * precision * recall) / (
        1 if (precision + recall) == 0 else (precision + recall)
    )
    return f1 * 100.0
