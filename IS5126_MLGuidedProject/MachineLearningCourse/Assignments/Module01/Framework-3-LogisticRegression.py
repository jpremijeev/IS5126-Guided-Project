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

runUnitTest = True  # False
if runUnitTest:
    # Little synthetic dataset to help with implementation. 2 features, 8 samples.
    xTrain = [
        [0.1, 0.1],
        [0.2, 0.2],
        [0.2, 0.1],
        [0.1, 0.2],
        [0.95, 0.95],
        [0.9, 0.8],
        [0.8, 0.9],
        [0.7, 0.6],
    ]
    yTrain = [0, 0, 0, 0, 1, 1, 1, 1]

    # create a linear model with the right number of weights initialized
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression

    model = LogisticRegression.LogisticRegression(featureCount=len(xTrain[0]))

    # To use this visualizer you need to install the PIL imaging library. Instructions are in the lecture notes.
    import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

    while not model.converged:
        # do 10 iterations of training
        model.incrementalFit(
            xTrain, yTrain, maxSteps=10, stepSize=1.0, convergence=0.005
        )

        # then look at the models weights
        model.visualize()

        # then look at how training set loss is converging
        print(
            " fit for %d iterations, train set loss is %.4f"
            % (model.totalGradientDescentSteps, model.loss(xTrain, yTrain))
        )

        # and visualize the model's decision boundary
        visualization = Visualize2D.Visualize2D(
            kOutputDirectory, "{0:04}.test".format(model.totalGradientDescentSteps)
        )
        visualization.Plot2DDataAndBinaryConcept(xTrain, yTrain, model)
        visualization.Save()

# Once your LogisticRegression learner seems to be working, set this flag to True and try it on the spam data
runSMSSpam = True  # True
if runSMSSpam:
    import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

    ############################
    # Set up the data

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

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=True)
    featurizer.CreateVocabulary(
        xTrainRaw, yTrain, supplementalVocabularyWords=["call", "to", "your"]
    )

    xTrain = featurizer.Featurize(xTrainRaw)
    xValidate = featurizer.Featurize(xValidateRaw)
    xTest = featurizer.Featurize(xTestRaw)

    #############################
    # Learn the logistic regression model

    print("Learning the logistic regression model:")
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression

    logisticRegressionModel = LogisticRegression.LogisticRegression()

    logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.005)

    #############################
    # Evaluate the model

    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    print("\nLogistic regression model:")
    logisticRegressionModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(
        yValidate,
        logisticRegressionModel.predict(xValidate, classificationThreshold=0.5),
    )

    # Tune the hyperparameter convergence by trying [0.01, 0.001, 0.0001, 0.00001] (with stepSize of 1.0)
    hyperparams = [0.01, 0.001, 0.0001, 0.00001]
    for p in hyperparams:
        logisticRegressionModel = LogisticRegression.LogisticRegression()
        logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=p)
        print("convergence parameter: ", p)
        EvaluateBinaryClassification.ExecuteAll(
            yValidate,
            logisticRegressionModel.predict(xValidate, classificationThreshold=0.5),
        )

    # convergence params  | steps to convergence  | validation set accuracy
    # 0.01                |   8                   |   0.84
    # 0.001               |   57                  |   0.87
    # 0.0001              |   186                 |   0.92
    # 0.00001             |   522                 |   0.93

    #################
    # You may find the following module helpful for making charts. You'll have to install matplotlib (see the lecture notes).
    #
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

    #
    # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates
    logisticRegressionModel = LogisticRegression.LogisticRegression()
    trainLosses, validationLosses = logisticRegressionModel.fit(
        xTrain,
        yTrain,
        stepSize=1.0,
        convergence=p,
        xValidation=xValidate,
        yValidation=yValidate,
    )
    # logisticRegressionModel = LogisticRegression.LogisticRegression()
    # validationLosses = logisticRegressionModel.fit(
    #     xValidate, yValidate, stepSize=1.0, convergence=p
    # )
    print(trainLosses, validationLosses)
    bigger = (
        trainLosses if len(trainLosses) > len(validationLosses) else validationLosses
    )
    smaller = trainLosses if bigger == validationLosses else validationLosses
    smaller += [smaller[-1]] * (len(bigger) - len(smaller))
    lossXLabels = [100 * x for x in range(1, len(bigger) + 1)]
    print("Generating chart")
    Charting.PlotSeries(
        [trainLosses, validationLosses],
        ["Train", "Validate"],
        lossXLabels,
        chartTitle="Logistic Regression",
        xAxisTitle="Gradient Descent Steps",
        yAxisTitle="Avg. Loss",
        outputDirectory=kOutputDirectory,
        fileName="3-Logistic Regression Train vs Validate loss",
    )
