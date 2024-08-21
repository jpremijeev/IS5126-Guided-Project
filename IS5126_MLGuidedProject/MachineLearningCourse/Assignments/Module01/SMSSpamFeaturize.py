import math


class SMSSpamFeaturize(object):
    """A class to coordinate turning SMS spam strings into feature vectors."""

    def __init__(self, useHandCraftedFeatures=False):
        # use hand crafted features specified in _FeaturizeXForHandCrafted()
        self.useHandCraftedFeatures = useHandCraftedFeatures

        self.ResetVocabulary()

    def ResetVocabulary(self):
        self.vocabularyCreated = False
        self.vocabulary = []

    def Tokenize(self, xRaw):
        return str.split(xRaw)

    def FindMostFrequentWords(self, x, n):
        # print("Stub FindMostFrequentWords in ", __file__)
        if n == 0:
            return []
        frequency = {}
        for data in x:
            tokens = list(set(self.Tokenize(data)))
            for token in tokens:
                if token not in frequency:
                    frequency[token] = 0
                frequency[token] += 1
        frequency_items = frequency.items()
        frequency_items = sorted(frequency_items, key=lambda x: x[1], reverse=True)
        nth_score = frequency_items[n - 1][1]
        # handle ties
        return list(
            map(lambda x: x[0], filter(lambda y: y[1] >= nth_score, frequency_items))
        )

    def FindTopWordsByMutualInformation(self, x, y, n):
        # print("Stub FindTopWordsByMutualInformation in ", __file__)
        if n == 0:
            return []
        frequency = {}
        for data, label in zip(x, y):
            tokens = list(set(self.Tokenize(data)))
            for token in tokens:
                if token not in frequency:
                    frequency[token] = {0: 0, 1: 0}
                frequency[token][label] += 1
        mutual_info = {}
        # total positive == (total y == 1)
        p_y_pos = (sum(y) + 1) / (len(y) + 2)
        # total negative = total data - total positive
        p_y_neg = (len(y) - sum(y) + 1) / (len(y) + 2)
        # total_words = sum(sum(list(frequency[x].values())) for x in frequency)
        # print(total_words)
        for key in frequency:
            x_pos = frequency[key][1]
            x_neg = frequency[key][0]
            p_xy_pos = (x_pos + 1) / (len(x) + 2)
            p_xy_neg = (x_neg + 1) / (len(x) + 2)
            # probability of x
            p_x = (x_pos + x_neg + 1) / (len(x) + 2)
            mutual_info_pos = p_xy_pos * math.log2(p_xy_pos / (p_x * p_y_pos))
            mutual_info_neg = p_xy_neg * math.log2(p_xy_neg / (p_x * p_y_neg))
            mutual_info[key] = mutual_info_neg + mutual_info_pos
        # print(mutual_info)
        mutual_info_items = mutual_info.items()
        mutual_info_items = sorted(mutual_info_items, key=lambda x: x[1], reverse=True)
        nth_score = mutual_info_items[n - 1][1]
        return list(
            map(lambda x: x[0], filter(lambda y: y[1] >= nth_score, mutual_info_items))
        )

    def CreateVocabulary(
        self,
        xTrainRaw,
        yTrainRaw,
        numFrequentWords=0,
        numMutualInformationWords=0,
        supplementalVocabularyWords=[],
    ):
        if self.vocabularyCreated:
            raise UserWarning(
                "Calling CreateVocabulary after the vocabulary was already created. Call ResetVocabulary to reinitialize."
            )

        # This function will eventually scan the strings in xTrain and choose which words to include in the vocabulary.
        #   But don't implement that until you reach the assignment that requires it...

        # For now, only use words that are passed in
        # self.vocabulary = self.vocabulary + supplementalVocabularyWords
        most_frequent = self.FindMostFrequentWords(xTrainRaw, numFrequentWords)
        mutual_info = self.FindTopWordsByMutualInformation(
            xTrainRaw, yTrainRaw, numMutualInformationWords
        )
        self.vocabulary = (
            self.vocabulary + supplementalVocabularyWords + most_frequent + mutual_info
        )

        self.vocabularyCreated = True

    def _FeaturizeXForVocabulary(self, xRaw):
        features = []

        # for each word in the vocabulary output a 1 if it appears in the SMS string, or a 0 if it does not
        tokens = self.Tokenize(xRaw)
        for word in self.vocabulary:
            if word in tokens:
                features.append(1)
            else:
                features.append(0)

        return features

    def _FeaturizeXForHandCraftedFeatures(self, xRaw):
        features = []

        # This function can produce an array of hand-crafted features to add on top of the vocabulary related features
        if self.useHandCraftedFeatures:
            # Have a feature for longer texts
            if len(xRaw) > 40:
                features.append(1)
            else:
                features.append(0)

            # Have a feature for texts with numbers in them
            if any(i.isdigit() for i in xRaw):
                features.append(1)
            else:
                features.append(0)

        return features

    def _FeatureizeX(self, xRaw):
        return self._FeaturizeXForVocabulary(
            xRaw
        ) + self._FeaturizeXForHandCraftedFeatures(xRaw)

    def Featurize(self, xSetRaw):
        return [self._FeatureizeX(x) for x in xSetRaw]

    def GetFeatureInfo(self, index):
        if index < len(self.vocabulary):
            return self.vocabulary[index]
        else:
            # return the zero based index of the heuristic feature
            return "Heuristic_%d" % (index - len(self.vocabulary))
