import time
import math
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate


class LogisticRegression(object):
    """Stub class for a Logistic Regression Model"""

    def __init__(self, featureCount=None):
        self.isInitialized = False

        if featureCount != None:
            self.__initialize(featureCount)

    def __testInput(self, x, y):
        if len(x) == 0:
            raise UserWarning("Trying to fit but can't fit on 0 training samples.")

        if len(x) != len(y):
            raise UserWarning("Trying to fit but length of x != length of y.")

    def __initialize(self, featureCount):
        self.weights = [0.0 for i in range(featureCount)]
        self.weight0 = 0.0

        self.converged = False
        self.totalGradientDescentSteps = 0

        self.isInitialized = True

    def loss(self, x, y):
        return EvaluateBinaryProbabilityEstimate.LogLoss(
            y, self.predictProbabilities(x)
        )

    def predictProbabilities(self, x):
        # For each sample do the dot product between features and weights (remember the bias weight, weight0)
        #  pass the results through the sigmoid function to convert to probabilities.
        dot_products = [
            self.weight0 + sum(w * f for w, f in zip(self.weights, curr_x))
            for curr_x in x
        ]  # w0 + w1x1 + w2x2 + ... + wnxn
        probability = [
            1 / (1 + math.exp(-product)) for product in dot_products
        ]  # sigmoid function
        # print("Stub predictProbabilities in ", __file__)
        return probability

    def predict(self, x, classificationThreshold=0.5):
        print("Stub predict in ", __file__)
        probabilities = self.predictProbabilities(x)
        classification = [
            1 if probability >= classificationThreshold else 0
            for probability in probabilities
        ]
        return classification

    def __gradientDescentStep(self, x, y, stepSize):
        self.totalGradientDescentSteps = self.totalGradientDescentSteps + 1

        gradient_w = [0.0] * len(self.weights)
        gradient_w0 = 0.0
        probs = self.predictProbabilities(x)
        errors = [ypred - ytrue for ypred, ytrue in zip(probs, y)]
        gradient_w0 = sum(errors)

        for i in range(len(x)):
            # error = ypred - ytrue
            # error = self.predictProbabilities([x[i]])[0] - y[i]
            # gradient_w0 += error
            for j in range(len(self.weights)):
                gradient_w[j] += errors[i] * x[i][j]

        # update to the opposite direction of the gradient
        for j in range(len(self.weights)):
            self.weights[j] -= stepSize * (gradient_w[j] / len(x))
        self.weight0 -= stepSize * (gradient_w0 / len(x))

        # print("Stub gradientDescentStep in ", __file__)

    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(
        self,
        x,
        y,
        maxSteps=1,
        stepSize=1.0,
        convergence=0.005,
        xValidation=None,
        yValidation=None,
    ):
        self.__testInput(x, y)
        if self.isInitialized == False:
            self.__initialize(len(x[0]))
        loss = []
        validation_loss = []
        total_losses = 0
        # do a maximum of 'maxSteps' of gradient descent with the indicated stepSize (use the __gradientDescentStep stub function for code clarity).
        #  stop and set self.converged to true if the mean log loss on the training set decreases by less than 'convergence' on a gradient descent step.
        prev_loss = None
        for step in range(maxSteps):
            self.__gradientDescentStep(x, y, stepSize)
            # print(prev_loss, self.loss(x, y))
            curr_loss = self.loss(x, y)
            if prev_loss and prev_loss - curr_loss < convergence:
                self.converged = True
                break
            prev_loss = curr_loss
            total_losses += curr_loss
            if (step + 1) % 100 == 0:
                if xValidation and yValidation:
                    validation_loss.append(self.loss(xValidation, yValidation))
                loss.append(total_losses / 100)
                total_losses = 0
        print("Stub incrementalFit in ", __file__)
        return loss, validation_loss

    def fit(
        self,
        x,
        y,
        maxSteps=50000,
        stepSize=1.0,
        convergence=0.005,
        verbose=True,
        xValidation=None,
        yValidation=None,
    ):
        startTime = time.time()

        loss, validation_loss = self.incrementalFit(
            x,
            y,
            maxSteps=maxSteps,
            stepSize=stepSize,
            convergence=convergence,
            xValidation=xValidation,
            yValidation=yValidation,
        )

        endTime = time.time()
        runtime = endTime - startTime

        if not self.converged:
            print(
                "Warning: did not converge after taking the maximum allowed number of steps."
            )
        elif verbose:
            print(
                "LogisticRegression converged in %d steps (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f."
                % (
                    self.totalGradientDescentSteps,
                    runtime,
                    len(self.weights),
                    stepSize,
                    convergence,
                )
            )
        return loss, validation_loss

    def visualize(self):
        print("w0: %f " % (self.weight0), end="")

        for i in range(len(self.weights)):
            print("w%d: %f " % (i + 1, self.weights[i]), end="")

        print("\n")
