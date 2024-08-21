# This file contains stubs for evaluating binary classifications. You must complete these functions as part of your assignment.
#     Each function takes in:
#           'y':           the arrary of 0/1 true class labels;
#           'yPredicted':  the prediction your model made for the cooresponding example.


def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if len(y) != len(yPredicted):
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again."
        )

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be 0 or 1."
        )


def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if y[i] == yPredicted[i]:
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct) / len(correct)


def Precision(y, yPredicted):
    print("Stub precision in ", __file__)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    if tp + fp == 0:  # to avoid zero division
        return 0.0
    precision = tp / (tp + fp)
    return precision


def Recall(y, yPredicted):
    print("Stub Recall in ", __file__)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    tp = confusion_matrix[1][1]
    fn = confusion_matrix[1][0]
    if tp + fn == 0:  # to avoid zero division
        return 0.0
    recall = tp / (tp + fn)
    return recall


def FalseNegativeRate(y, yPredicted):
    print("Stub FalseNegativeRate in ", __file__)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    tp = confusion_matrix[1][1]
    fn = confusion_matrix[1][0]
    if tp + fn == 0:  # to avoid zero division
        return 0.0
    fnr = fn / (tp + fn)
    return fnr


def FalsePositiveRate(y, yPredicted):
    print("Stub FalsePositiveRate in ", __file__)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    if tn + fp == 0:  # to avoid zero division
        return 0.0
    fpr = fp / (tn + fp)
    return fpr


def ConfusionMatrix(y, yPredicted):
    # This function should return: [[<# True Negatives>, <# False Positives>], [<# False Negatives>, <# True Positives>]]
    #  Hint: writing this function first might make the others easier...
    tn, fp, fn, tp = 0, 0, 0, 0

    for true_label, predicted_label in zip(y, yPredicted):
        if true_label == 0 and predicted_label == 0:  # true negative
            tn += 1
        elif (
            true_label == 0 and predicted_label == 1
        ):  # false positive, predicted positive but actually negative
            fp += 1
        elif (
            true_label == 1 and predicted_label == 0
        ):  # false negative, predicted negative but actually positive
            fn += 1
        elif true_label == 1 and predicted_label == 1:  # true positive
            tp += 1

    print("Stub preConfusionMatrix in ", __file__)
    return [[tn, fp], [fn, tp]]


def ExecuteAll(y, yPredicted):
    print("Confusion Matrix:", ConfusionMatrix(y, yPredicted))
    print("Accuracy: %.2f" % (Accuracy(y, yPredicted)))
    print("Precision: %.2f" % (Precision(y, yPredicted)))
    print("Recall: %.2f" % (Recall(y, yPredicted)))
    print("FPR: %.2f" % (FalsePositiveRate(y, yPredicted)))
    print("FNR: %.2f" % (FalseNegativeRate(y, yPredicted)))
