from configparser import ConfigParser


class ConfigManager:
    """Manages the configuration of the training and testing stages"""
    def __init__(self, filePath):
        parser = ConfigParser()
        read = parser.read(filePath)

        # try to open config file
        if not read:
            raise ValueError('Could not open file ' + filePath)

        # display sections from config file
        print(parser.sections())

        self.training = TrainingConfig(parser['TRAINING'])
        self.testing = TestingConfig(parser['TESTING'])


class TrainingConfig:
    """Initialises values for training the model"""
    def __init__(self, trainSection):
        self.batchSize = int(trainSection['batchSize'])
        self.imgHeight = int(trainSection['imgHeight'])
        self.imgHeight = int(trainSection['imgHeight'])
        self.imgChannels = int(trainSection['imgChannels'])
        self.maskChannels = int(trainSection['maskChannels'])
        self.numEpochs = int(trainSection['numEpochs'])
        self.genLR = float(trainSection['genLR'])
        self.discLR = float(trainSection['discLR'])
        self.genFeatures = int(trainSection['genFeatures'])
        self.discFeatures = int(trainSection['discFeatures'])
        self.currentEpoch = int(trainSection['currentEpoch'])
        self.sampleInterval = int(trainSection['sampleInterval'])
        self.beta1 = float(trainSection['beta1'])
        self.beta2 = float(trainSection['beta2'])
        self.epsilon = float(trainSection['epsilon'])
        self.lastTrainedEpoch = int(trainSection['lastTrainedEpoch'])


class TestingConfig:
    """Initialises values for testing the model"""
    def __init__(self, testSection):
        self.batchSize = int(testSection['batchSize'])
        self.imgHeight = int(testSection['imgHeight'])
        self.imgHeight = int(testSection['imgHeight'])
        self.imgChannels = int(testSection['imgChannels'])
        self.maskChannels = int(testSection['maskChannels'])
        self.numEpochs = int(testSection['numEpochs'])
        self.genLR = float(testSection['genLR'])
        self.discLR = float(testSection['discLR'])
        self.genFeatures = int(testSection['genFeatures'])
        self.discFeatures = int(testSection['discFeatures'])
        self.currentEpoch = int(testSection['currentEpoch'])
        self.sampleInterval = int(testSection['sampleInterval'])
        self.beta1 = float(testSection['beta1'])
        self.beta2 = float(testSection['beta2'])
        self.epsilon = float(testSection['epsilon'])
        self.lastTrainedEpoch = int(testSection['lastTrainedEpoch'])
        self.lastImg = int(testSection['lastImg'])
