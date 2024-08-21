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

(
    xTrainRaw,
    yTrain,
    xValidateRaw,
    yValidate,
    xTestRaw,
    yTest,
) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=0.1, percentTest=0.1)

import MachineLearningCourse.Assignments.Module01.SMSSpamFeaturize as SMSSpamFeaturize

findTop10Words = True
if findTop10Words:
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)

    print(
        "Top 10 words by frequency: ", featurizer.FindMostFrequentWords(xTrainRaw, 10)
    )
    print(
        "Top 10 words by mutual information: ",
        featurizer.FindTopWordsByMutualInformation(xTrainRaw, yTrain, 10),
    )

# set to true when your implementation of the 'FindWords' part of the assignment is working
doModeling = True  # True
if doModeling:
    # Now get into model training
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel

    # The hyperparameters to use with logistic regression for this assignment
    stepSize = 1.0
    convergence = 0.001

    # Remeber to create a new featurizer object/vocabulary for each part of the assignment
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=10)

    # Remember to reprocess the raw data whenever you change the featurizer
    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest = featurizer.Featurize(xTestRaw)

    ## Good luck!
    logisticRegressionModel = LogisticRegression.LogisticRegression()
    logisticRegressionModel.fit(
        xTrain, yTrain, stepSize=stepSize, convergence=convergence
    )
    yPredicted = logisticRegressionModel.predict(xValidate, classificationThreshold=0.5)
    EvaluateBinaryClassification.ExecuteAll(
        yValidate,
        yPredicted,
    )
    # for 4b
    compare = True
    if compare:
        num_freq_word = 10
        acc = EvaluateBinaryClassification.Accuracy(yValidate, yPredicted)
        common_model = MostCommonClassModel.MostCommonClassModel()
        common_model.fit(xTrainRaw, yTrain)
        y_val_predicted = common_model.predict(xValidateRaw)
        common_acc = EvaluateBinaryClassification.Accuracy(yValidate, y_val_predicted)
        print(
            f"Using 10 most freq words: logreg acc: {acc}, most common class acc: {common_acc}"
        )
        while acc <= common_acc:  # most common class accuracy
            featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
            featurizer.CreateVocabulary(
                xTrainRaw, yTrain, numFrequentWords=num_freq_word
            )

            # Remember to reprocess the raw data whenever you change the featurizer
            xTrain = featurizer.Featurize(xTrainRaw)
            xValidate = featurizer.Featurize(xValidateRaw)
            logisticRegressionModel = LogisticRegression.LogisticRegression()
            logisticRegressionModel.fit(
                xTrain, yTrain, stepSize=stepSize, convergence=convergence
            )
            yPredicted = logisticRegressionModel.predict(
                xValidate, classificationThreshold=0.5
            )
            acc = EvaluateBinaryClassification.Accuracy(yValidate, yPredicted)
            if acc > common_acc:
                print(
                    f"Using {num_freq_word} frequent words will give better result than the most common class predictor, with accuracy {acc}"
                )
                break
            num_freq_word += 5
    # for 4c
    val_losses = []
    train_losses = []
    numFeatures = [1, 10, 20, 30, 40, 50]
    for n in numFeatures:
        print("Using " + str(n) + " most freq words")
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords=n)
        xTrain = featurizer.Featurize(xTrainRaw)
        xValidate = featurizer.Featurize(xValidateRaw)
        xTest = featurizer.Featurize(xTestRaw)
        logisticRegressionModel = LogisticRegression.LogisticRegression()
        logisticRegressionModel.fit(
            xTrain, yTrain, stepSize=stepSize, convergence=convergence
        )
        train_losses.append(logisticRegressionModel.loss(xTrain, yTrain))
        val_losses.append(logisticRegressionModel.loss(xValidate, yValidate))
    # plot here
    Charting.PlotTrainValidateTestSeries(
        train_losses,
        val_losses,
        xAxisPoints=numFeatures,
        chartTitle="Num of Frequent Words vs Validation Loss",
        xAxisTitle="Number of Features",
        yAxisTitle="Losses",
        outputDirectory=kOutputDirectory,
        fileName="4 - Num Frequent Words vs Validation Loss",
    )

    train_losses = []
    val_losses = []
    for n in numFeatures:
        print("Using " + str(n) + " mutual info")
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords=n)
        xTrain = featurizer.Featurize(xTrainRaw)
        xValidate = featurizer.Featurize(xValidateRaw)
        xTest = featurizer.Featurize(xTestRaw)
        logisticRegressionModel = LogisticRegression.LogisticRegression()
        logisticRegressionModel.fit(
            xTrain, yTrain, stepSize=stepSize, convergence=convergence
        )
        train_losses.append(logisticRegressionModel.loss(xTrain, yTrain))
        val_losses.append(logisticRegressionModel.loss(xValidate, yValidate))
    # plot here
    Charting.PlotTrainValidateTestSeries(
        train_losses,
        val_losses,
        xAxisPoints=numFeatures,
        chartTitle="Num of Mutual Information vs Validation Loss",
        xAxisTitle="Number of Features",
        yAxisTitle="Losses",
        outputDirectory=kOutputDirectory,
        fileName="4 - Num of Top Mutual Information vs Validation Loss",
    )
