# This preamble tells the Python interpreter to look in the folder containing
# the MachineLearningCourse dir for the relevant Python files.
import sys, os

curDir = os.path.dirname(os.path.abspath(__file__))
projDir = os.path.join(curDir, "..", "..", "..")
sys.path.append(projDir)  # look in the directory containing MachineLearningCourse/
sys.path.append(curDir)  # look in the directory of this file too, i.e., Module01/

# specify the directory to store your visualization files
kOutputDirectory = "C:\\Users\\Premi\\Desktop\\Semester 3\\IS5126 Hands-on with Applied Analytics\\Guided Project IS5126\\IS5126_MLGuidedProject\\MachineLearningCourse\\Visualizations"
# kOutputDirectory = "C:\\temp\\visualize" #use this for Windows

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = (
    Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=0.1, percentTest=0.1)
)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation

import time


# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
    pointsToEvaluate = 100
    thresholds = [x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
    FPRs = []
    FNRs = []
    yPredicted = model.predictProbabilities(xValidate)

    try:
        for threshold in thresholds:
            yHats = [1 if pred > threshold else 0 for pred in yPredicted]
            FPRs.append(
                EvaluateBinaryClassification.FalsePositiveRate(yValidate, yHats)
            )
            FNRs.append(
                EvaluateBinaryClassification.FalseNegativeRate(yValidate, yHats)
            )
    except NotImplementedError:
        raise UserWarning(
            "The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications."
        )

    return (FPRs, FNRs, thresholds)


import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

## This function will help you plot with error bars. Use it just like PlotSeries, but with parallel arrays of error bar sizes in the second variable
#     note that the error bar size is drawn above and below the series value. So if the series value is .8 and the confidence interval is .78 - .82, then the value to use for the error bar is .02

# Charting.PlotSeriesWithErrorBars([series1, series2], [errorBarsForSeries1, errorBarsForSeries2], ["Series1", "Series2"], xValues, chartTitle="<>", xAxisTitle="<>", yAxisTitle="<>", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="<name>")


## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
import math

best_params = {
    "numMutualInformationWords": {
        "validation": 0,
        "best": 0,
        "best_value": 0,
        "time": 0,
    },
    "stepSize": {"validation": 0, "best": 0, "best_value": 0, "time": 0},
    "convergence": {"validation": 0, "best": 0, "best_value": 0, "time": 0},
    "numFrequentWords": {"validation": 0, "best": 0, "best_value": 0, "time": 0},
}


def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds=5):
    startTime = time.time()
    param_to_optimize = runSpecification["optimizing"]
    print(f"Using {runSpecification[param_to_optimize]} for {param_to_optimize}")

    # HERE upgrade this to use crossvalidation
    fold_size = math.floor(len(xTrainRaw) / numberOfFolds)
    folds = []
    accuracies = []

    for x in range(numberOfFolds):
        start_idx = x * fold_size
        end_idx = start_idx + fold_size
        validate_x = xTrainRaw[start_idx:end_idx]
        validate_y = yTrain[start_idx:end_idx]
        train_x = xTrainRaw[:start_idx] + xTrainRaw[end_idx:]
        train_y = yTrain[:start_idx] + yTrain[end_idx:]
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(
            xTrainRaw,
            yTrain,
            numFrequentWords=runSpecification["numFrequentWords"],
            numMutualInformationWords=runSpecification["numMutualInformationWords"],
        )
        x_train = featurizer.Featurize(train_x)
        x_validate = featurizer.Featurize(validate_x)
        model = LogisticRegression.LogisticRegression()
        model.fit(
            x_train,
            train_y,
            convergence=runSpecification["convergence"],
            stepSize=runSpecification["stepSize"],
            verbose=True,
        )
        validation_accuracy = EvaluateBinaryClassification.Accuracy(
            validate_y, model.predict(x_validate)
        )
        accuracies.append(validation_accuracy)

    accuracy = sum(accuracies) / len(accuracies)
    runSpecification["accuracy"] = accuracy
    std = (sum((acc - accuracy) ** 2 for acc in accuracies) / len(accuracies)) ** (
        1 / 2
    )
    runSpecification["error_bound"] = (
        0.67 * std / (len(accuracies) ** (1 / 2))
    )  # z score for 0.5 2 sided is 0.67
    upper_75 = (
        0.67 * std / (len(accuracies) ** (1 / 2))
    ) + accuracy  # z score for 0.75 1 sided is 0.67

    # runSpecification["accuracy"] = validationSetAccuracy
    # HERE upgrade this to calculate and save some error bounds...

    endTime = time.time()
    runSpecification["runtime"] = endTime - startTime
    best_val = best_params[param_to_optimize]["best_value"]
    best_time = best_params[param_to_optimize]["time"]

    # the value you pick is tied with the highest accuracy value according to a 75% 1-sided bound (or beats all others according to this bound), and
    # the value you pick has the lowest runtime among these ‘tied’ values.
    if upper_75 > best_val or (
        upper_75 == best_val and runSpecification["runtime"] < best_time
    ):
        best_params[param_to_optimize]["best_value"] = upper_75
        best_params[param_to_optimize]["best"] = runSpecification[param_to_optimize]
        best_params[param_to_optimize]["time"] = runSpecification["runtime"]

    return runSpecification


validation_accuracies = []
seriesFPRs = []
seriesFNRs = []
seriesLabels = []


# helper function to evaluate test data
def evaluate_test(params, model_desc):
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    x_train_full_raw = xTrainRaw + xValidateRaw  # to test with test data
    y_train_full = yTrain + yValidate
    featurizer.CreateVocabulary(
        x_train_full_raw,
        y_train_full,
        numFrequentWords=params["numFrequentWords"],
        numMutualInformationWords=params["numMutualInformationWords"],
    )

    x_train_full = featurizer.Featurize(x_train_full_raw)
    x_test = featurizer.Featurize(xTestRaw)

    model = LogisticRegression.LogisticRegression()
    model.fit(
        x_train_full,
        y_train_full,
        convergence=params["convergence"],
        stepSize=params["stepSize"],
        verbose=True,
    )

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(
        model, x_test, yTest
    )
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append(model_desc)


def run_hyperparameter_tuning(hyper_params, default_params, verbose=True):
    for param in hyper_params:
        print(f"Running hyperparameters tuning for {param}")
        evaluationRunSpecifications = []
        values = hyper_params[param]
        for value in values:
            runSpecification = default_params.copy()
            runSpecification["optimizing"] = param
            runSpecification[param] = value
            evaluationRunSpecifications.append(runSpecification)
        ## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
        # from joblib import Parallel, delayed
        # evaluations = Parallel(n_jobs=12)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)
        evaluations = [
            ExecuteEvaluationRun(runSpec, xTrainRaw, yTrain, numberOfFolds=2)
            for runSpec in evaluationRunSpecifications
        ]
        # After each sweep, update the hyperparameter value
        default_params[param] = best_params[param]["best"]

        series = list(map(lambda x: x["accuracy"], evaluationRunSpecifications))
        run_times = list(map(lambda x: x["runtime"], evaluationRunSpecifications))
        error_bars = list(map(lambda x: x["error_bound"], evaluationRunSpecifications))
        """
        The errorbar sizes:
        scalar: Symmetric +/- values for all data points.
        shape(N,): Symmetric +/-values for each data point.
        shape(2, N): Separate - and + values for each bar. First row contains the lower errors, the second row contains the upper errors.
        """
        if verbose:
            print(f"Generating chart for {param}")
            Charting.PlotSeriesWithErrorBars(
                [series],
                [error_bars],
                ["Accuracy"],
                values,
                chartTitle=f"{param} Hyperparameter Tuning",
                xAxisTitle=param,
                yAxisTitle="Accuracy",
                yBotLimit=0.8,
                outputDirectory=kOutputDirectory,
                fileName=f"7-{param}",
            )
            Charting.PlotSeries(
                [run_times],
                ["Run times"],
                values,
                chartTitle=f"{param} Hyperparameter Tuning Runtime",
                xAxisTitle=param,
                yAxisTitle="Runtime",
                outputDirectory=kOutputDirectory,
                fileName=f"7-{param}-runtime",
            )

        for evaluation in evaluations:
            print(evaluation)

        print(f"Using {default_params} for validation")
        # After optimizing each hyperparameter, evaluate the accuracy on the validation set
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(
            xTrainRaw,
            yTrain,
            numFrequentWords=default_params["numFrequentWords"],
            numMutualInformationWords=default_params["numMutualInformationWords"],
        )

        xTrain = featurizer.Featurize(xTrainRaw)
        xValidate = featurizer.Featurize(xValidateRaw)

        model = LogisticRegression.LogisticRegression()
        model.fit(
            xTrain,
            yTrain,
            convergence=default_params["convergence"],
            stepSize=default_params["stepSize"],
            verbose=True,
        )

        validationSetAccuracy = EvaluateBinaryClassification.Accuracy(
            yValidate, model.predict(xValidate)
        )

        validation_accuracies.append(validationSetAccuracy)

    print(f"Final params: {default_params}, accuracy: {validationSetAccuracy}")
    return default_params


default_params = {
    "numMutualInformationWords": 20,
    "stepSize": 1.0,
    "convergence": 0.005,
    "numFrequentWords": 0,
}
chosen_params = default_params.copy()

# evaluate initial model
print(f"Evaluating on test set using {default_params}")
evaluate_test(default_params, "Base model with default params")

hyperparameter_tuning = True
if hyperparameter_tuning:
    hyper_params = {
        "numMutualInformationWords": [0, 20, 80, 140, 200, 260, 320],
        "numFrequentWords": [0, 60, 120, 180, 240, 300, 360],
        "stepSize": [0.05, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "convergence": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    }
    chosen_params = run_hyperparameter_tuning(hyper_params, chosen_params.copy())

second_round = True
if second_round:
    hyper_params = {
        "numMutualInformationWords": [340, 360],
        "stepSize": [3.7, 4],
    }
    chosen_params = run_hyperparameter_tuning(
        hyper_params, chosen_params.copy(), verbose=False
    )

if hyperparameter_tuning:
    chosen_params = chosen_params.copy()
else:
    chosen_params = {
        "numMutualInformationWords": 360,
        "stepSize": 4,
        "convergence": 0.0001,
        "numFrequentWords": 360,
    }

# plot step 4
mean_acc = sum(validation_accuracies) / len(validation_accuracies)
val_std = (
    sum((acc - mean_acc) ** 2 for acc in validation_accuracies)
    / len(validation_accuracies)
) ** (1 / 2)
validation_errors = list(
    map(
        lambda x: 0.64 * val_std / (len(validation_accuracies) ** (1 / 2)),
        validation_accuracies,
    )
)
Charting.PlotSeriesWithErrorBars(
    [validation_accuracies],
    [validation_errors],
    ["Accuracy"],
    list(range(1, 7)),
    chartTitle=f"Hyperparameter Tuning Validation Accuracy",
    xAxisTitle="# sweep",
    yAxisTitle="Accuracy",
    yBotLimit=0.8,
    outputDirectory=kOutputDirectory,
    fileName=f"7-validation",
)

# part d
print(f"Evaluating on test set using {chosen_params}")
evaluate_test(chosen_params, "Better model with hyperparameter tuning")

Charting.PlotROCs(
    seriesFPRs,
    seriesFNRs,
    seriesLabels,
    useLines=True,
    chartTitle="ROC Comparison",
    xAxisTitle="False Negative Rate",
    yAxisTitle="False Positive Rate",
    outputDirectory=kOutputDirectory,
    fileName="7-ROC Hyperparameter Tuning",
)

# Good luck!
# thanks
