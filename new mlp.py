import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import texttable as tt

# Read data file
dataFile = pd.read_excel('Data.xlsx', sheet_name='Readable data')

# Get inputs and outputs from file
inputs = np.array(
    (dataFile['Month'], dataFile['Crakehill'], dataFile['Skip Bridge'], dataFile['Westwick'], dataFile['Skelton'], dataFile['Arkengarthdale'], dataFile['East Cowton'], dataFile['Malham Tarn'], dataFile['Snaizeholme']), dtype=float).T
outputs = np.array(([dataFile['Output']]), dtype=float).T

rows = inputs.shape[0]

# Split dataset
# Split inputs
trainingInputs = inputs[:round(rows*0.6), :]
trainingValidationInputs = inputs[:trainingInputs.shape[0]+round(rows*0.2):, :]
validationInputs = inputs[trainingInputs.shape[0]
    :trainingInputs.shape[0]+round(rows*0.2):, :]
testInputs = inputs[trainingInputs.shape[0]+validationInputs.shape[0]:, :]
# Split outputs
trainingOutputs = outputs[:round(rows*0.6), :]
trainingValidationOutputs = outputs[:
                                    trainingOutputs.shape[0]+round(rows*0.2):, :]
validationOutputs = outputs[trainingOutputs.shape[0]
    :trainingOutputs.shape[0]+round(rows*0.2):, :]
testOutputs = outputs[trainingOutputs.shape[0]+validationOutputs.shape[0]:, :]

print("Training data size:", trainingInputs.shape[0])
print("Training and validation data size:", trainingValidationOutputs.shape[0])
print("Validation data size:", validationInputs.shape[0])
print("Test data size:", testInputs.shape[0])

# Output dimensions of arrays
print(inputs.shape)
print(outputs.shape)

# inputsMax = np.amax(inputs, axis=0)
# Get max values of inputs and outputs (separate for each row as it makes it more accurate)
inputsMax = np.amax(trainingValidationInputs, axis=0)
inputsMin = np.amin(trainingValidationInputs, axis=0)

outputsMax = np.amax(trainingValidationOutputs, axis=0)
outputsMin = np.amin(trainingValidationOutputs, axis=0)

print("inputsMax:", inputsMax)
print("inputsMin:", inputsMin)
print("outputsMax:", outputsMax)
print("outputsMin:", outputsMin)


def standardise(n, min, max):
    """
    Standardise n to 0.1,0.9 range using the min and max specified
    """
    return 0.8*((n-min)/(max-min))+0.1


# Standardise inputs and outputs
# Standardise inputs
inputs = standardise(inputs, inputsMin, inputsMax)
trainingInputs = standardise(trainingInputs, inputsMin, inputsMax)
validationInputs = standardise(validationInputs, inputsMin, inputsMax)
testInputs = standardise(testInputs, inputsMin, inputsMax)
# Standardise outputs
outputs = standardise(outputs, outputsMin, outputsMax)
trainingOutputs = standardise(trainingOutputs, outputsMin, outputsMax)
validationOutputs = standardise(validationOutputs, outputsMin, outputsMax)
testOutputs = standardise(testOutputs, outputsMin, outputsMax)

# Open output excel file
writer = pd.ExcelWriter("output.xlsx", engine="xlsxwriter")


class NeuralNetwork:
    def __init__(self, stepSize, randSeed=1, hiddenNodes=7, outputNodes=1):
        """
            Initialise Neural Network, stepSize, hiddenNodes and outputNodes
            can be configured during this stage. Attributes that will be used
            in other parts of the algorithm are declared here
        """
        # Configure Neural Network learning settings
        self.randSeed = randSeed
        self.stepSize = stepSize
        self.momentumRate = 0.9
        self.boldDriverConstant = 4  # (Percent)

        # Congigure Neural network size (number of nodes)
        self.inputNodes = trainingInputs.shape[1]
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        # Generate random weights for the input and hidden layer nodes
        np.random.seed(randSeed)
        self.inputWeights = np.random.randn(self.inputNodes, self.hiddenNodes)
        self.hiddenWeights = np.random.randn(
            self.hiddenNodes, self.outputNodes)

        # Used to store the best input and hidden weights when avoiding overtraining
        self.bestInputWeights = self.inputWeights.copy()
        self.bestHiddenWeights = self.hiddenWeights.copy()

        # Stores the weight change to be used with the momentum
        self.prevInputChange = np.zeros(self.inputWeights.shape)
        self.prevHiddenChange = np.zeros(self.hiddenWeights.shape)

        # Set initial error values
        self.prevError = np.array([])

    def destandardise(self, n):
        """
        Destandardise value to 0.1,0.9 range using min and max outputs
        """
        return ((n-0.1)/0.8)*(outputsMax-outputsMin)+outputsMin

    def sigmoid(self, n):
        """
        Sigmoid activation function for calculating error
        """
        return 1/(1 + np.exp(-n))

    def derived_sigmoid(self, n):
        """
        Derived Sigmoid activation function
        """
        return n * (1 - n)

    def forwardPass(self, inputs, destandardiseOutput=False):
        """
        Forward pass through the network, calculate modelled value using
        weights and values of the nodes
        """
        # Sj: Dot product of inputs and input node weights
        self.weightedInputs = np.dot(inputs, self.inputWeights)
        # f(Sj): Sigmoid function on dot product
        self.activatedWeightedInputs = self.sigmoid(self.weightedInputs)
        # Sj: Dot product of hidden node biases and hidden node weights
        self.weightedHidden = np.dot(
            self.activatedWeightedInputs, self.hiddenWeights)
        # f(Sj): Sigmoid function on dot product
        activatedWeightedHidden = self.sigmoid(self.weightedHidden)
        # If output should be destandardised
        if(destandardiseOutput):
            activatedWeightedHidden = self.destandardise(
                activatedWeightedHidden)
        return activatedWeightedHidden

    def backwardPass(self, inputs, outputs, output, momentum, boldDriver):
        """
        Backward pass through the network, update weights. Depending on
        improvements that have been specified, results will differ
        """
        # Calculate error
        self.outputError = outputs - output
        # Get delta value of error
        self.delta = (self.outputError) * self.derived_sigmoid(output)

        # Calculate error of weights inputs
        self.activatedWeightedInputsError = self.delta.dot(
            self.hiddenWeights.T)

        # If bold driver improvement is activated
        if(boldDriver):
            # Get delta value of weighted input error
            if(len(self.prevError) > 0):
                percentageChange = ((np.mean(np.square(self.outputError)) - np.mean(
                    np.square(self.prevError)))/abs(np.mean(np.square(self.prevError))))*100
                # Error function increased by boldDriverConstant percent
                if(abs(percentageChange) > self.boldDriverConstant):
                    if(self.stepSize > 0.01):
                        # Undo weight change
                        self.inputWeights -= self.prevInputChange
                        self.hiddenWeights -= self.prevHiddenChange
                        # Decrease step size
                        self.stepSize *= 0.7
                else:
                    # Increase step size
                    if(self.stepSize < 0.5/1.05):
                        self.stepSize *= 1.05

        # Calculate delta of activated weighted inputs
        self.activatedWeightedInputsDelta = self.activatedWeightedInputsError * \
            self.derived_sigmoid(self.activatedWeightedInputs)

        # Update weights of input nodes
        self.prevInputChange = inputWeightChange = self.stepSize * \
            inputs.T.dot(self.activatedWeightedInputsDelta)
        self.inputWeights += inputWeightChange
        # If momentum improvement is being used
        if (momentum):
            self.inputWeights += self.momentumRate*abs(self.prevInputChange)

        # Update weights of hidden nodes
        self.prevHiddenChange = hiddenWeightChange = self.stepSize * \
            self.activatedWeightedInputs.T.dot(self.delta)
        self.hiddenWeights += hiddenWeightChange
        # If momentum improvement is being used
        if (momentum):
            self.hiddenWeights += self.momentumRate*abs(self.prevHiddenChange)

        self.prevError = self.outputError

    def train(self, inputs, outputs, momentum, boldDriver):
        """
        Run a single epoch of training
        """
        # Calculate modelled output using a forward pass
        output = self.forwardPass(inputs)
        # Update weights using backward pass based on forward pass results
        self.backwardPass(inputs, outputs, output, momentum, boldDriver)

    # ====== Model assessment methods ======
    def RSME(self, modelledValues, observedValues):
        """
        Calculate Root squared mean error
        """
        return np.sqrt(sum(((modelledValues-observedValues)**2)/len(modelledValues)))

    def MSRE(self, modelledValues, observedValues):
        """
        Calculate Mean squared relative error
        """
        return ((sum(((modelledValues-observedValues)/observedValues)**2))/len(modelledValues))[0]

    def CE(self, modelledValues, observedValues):
        """ 
        Calculate Coefficient of efficiency
        """
        return (1-(sum((modelledValues-observedValues)**2)/sum((observedValues-observedValues.mean())**2)))[0]

    def RSqr(self, modelledValues, observedValues):
        """
        Calculate R-Squared (Coefficient of determination)
        """
        return (((sum((observedValues-observedValues.mean())*(modelledValues-modelledValues.mean()))) /
                 (np.sqrt(sum((observedValues-observedValues.mean())**2)*sum((modelledValues-modelledValues.mean())**2))))**2)[0]

    def run(self, epochs=100000, minMSE=0, momentum=False, boldDriver=False, annealing=False, showIncrements=True, showTable=True, showGraph=False, outputToExcel=False):
        """
        Run the Neural Network with specified configurations 
        #########################
        ## BACK PROP ALGORITHM ##
        #########################
        """
        # For graph plotting
        trainingYPoints = []
        validationYPoints = []
        XPoints = []

        # Get activated improvements
        improvements = []
        for improvement in zip(["Momentum", "Bold Driver", "Annealing"], [momentum, boldDriver, annealing]):
            if(improvement[1]):
                improvements.append(improvement[0])

        # For storing the best epoch and lowest MSE
        bestEpoch = 0
        minTrainValidMSE = 2

        # Cycle through each epoch
        for i in range(epochs):
            # If minumum MSE has been met, exit loop
            if(np.mean(np.square(trainingOutputs - self.forwardPass(trainingInputs))) < minMSE):
                break
            # If annealing improvement is activated
            if(annealing):
                self.stepSize = 0.01+(0.1-0.01) * \
                    (1-(1/(1+np.exp(10-(20*i)/epochs))))

            # Print loss every 100 epochs
            if (i % 100 == 0):
                # Finds the point in training where the sum of the training and validation data errors is the lowest (prevents overtraining)
                if(np.mean(np.square(validationOutputs - self.forwardPass(validationInputs))) + np.mean(np.square(trainingOutputs - self.forwardPass(trainingInputs))) < minTrainValidMSE):
                    minTrainValidMSE = np.mean(np.square(validationOutputs - self.forwardPass(validationInputs, False))) + np.mean(
                        np.square(trainingOutputs - self.forwardPass(trainingInputs, False)))
                    bestEpoch = i
                    self.bestInputWeights = self.inputWeights.copy()
                    self.bestHiddenWeights = self.hiddenWeights.copy()

                # For plotting on the graph
                XPoints.append(i)
                trainingYPoints.append(
                    np.mean(np.square(trainingOutputs - self.forwardPass(trainingInputs))))
                validationYPoints.append(
                    np.mean(np.square(validationOutputs - self.forwardPass(validationInputs))))

                # If show increments is specified, output training and validation MSE for every 100 epochs
                if(showIncrements):
                    print("Training MSE at", i, "epochs:",
                          round(np.mean(np.square(trainingOutputs -
                                self.forwardPass(trainingInputs, False))), 6),
                          "\tvalidation MSE:", round(np.mean(np.square(validationOutputs - self.forwardPass(validationInputs, False))), 6))

            # Train network
            self.train(trainingInputs, trainingOutputs, momentum, boldDriver)

        # Output model assessment methods on the data
        print("\n###########################")
        print(" Neural Network Assessment ")
        print("---------------------------")
        print(" Improvements:", ' '.join(improvements))
        print("###########################")
        print("Best epoch", bestEpoch)
        print("Final combined MSE:", np.mean(np.square(validationOutputs - self.forwardPass(validationInputs))
                                             ) + np.mean(np.square(trainingOutputs - self.forwardPass(trainingInputs))))
        print("Min combined MSE:", minTrainValidMSE)
        print("Final step size: ", self.stepSize)

        # Set input and hidden weights to the best that were found during training
        self.inputWeights = self.bestInputWeights
        self.hiddenWeights = self.bestHiddenWeights

        # For outputting table to console
        textTable = tt.Texttable()
        textTable.add_rows(
            [['Data', 'Standardised', 'MSE', 'RSME', 'MSRE', 'CE', 'RSqr']])
        # Add model assessment results to table
        for data in zip(["Training", "Validation", "Testing"], [trainingInputs, validationInputs, testInputs], [trainingOutputs, validationOutputs, testOutputs]):
            # Standardised
            textTable.add_row([data[0], 'True',
                               np.mean(
                                   np.square(data[2] - self.forwardPass(data[1]))),
                               self.RSME(self.forwardPass(data[1]), data[2]),
                               self.MSRE(self.forwardPass(data[1]), data[2]),
                               self.CE(self.forwardPass(data[1]), data[2]),
                               self.RSqr(self.forwardPass(data[1]), data[2])])
            # Destandardised
            textTable.add_row([data[0], 'False',
                               np.mean(np.square(self.destandardise(
                                   data[2]) - self.forwardPass(data[1], True))),
                               self.RSME(self.forwardPass(
                                   data[1], True), self.destandardise(data[2])),
                               self.MSRE(self.forwardPass(
                                   data[1], True), self.destandardise(data[2])),
                               self.CE(self.forwardPass(
                                   data[1], True), self.destandardise(data[2])),
                               self.RSqr(self.forwardPass(data[1], True), self.destandardise(data[2]))])
        # If show table is specified, output to console
        if(showTable):
            print(textTable.draw())

        # Configure graph
        plt.title("Training and Validation MSE's over the training period")
        plt.plot(XPoints, trainingYPoints, label="Training MSE")
        plt.plot(XPoints, validationYPoints, label="Validation MSE")
        plt.axvline(x=bestEpoch, label="Lowest T+V MSE", linestyle="--")
        plt.ylabel("MSE")
        plt.xlabel("Epochs")
        plt.legend()
        # If show graph is specified, output
        if(showGraph):
            plt.show()
        # Clear graph (in case another network is being trained after)
        plt.cla()
        # If output to excel is specified, add sheet to output file with data
        if(outputToExcel):
            outputData = pd.DataFrame(zip(XPoints, trainingYPoints, validationYPoints), columns=[
                                      "Epochs", "Training MSE", "Validation MSE"])
            outputData.to_excel(writer, sheet_name="{} e-{} s-{}.xlsx".format(
                ' '.join(improvements), epochs, self.randSeed), index=False)


# No improvements
NN1 = NeuralNetwork(0.03, randSeed=1)
NN1.run(epochs=100000, showIncrements=False,
        showTable=True, showGraph=False, outputToExcel=True)
# Momentum
NN2 = NeuralNetwork(0.04, randSeed=1)
NN2.run(epochs=100000, momentum=True, showIncrements=False,
        showTable=True, showGraph=False, outputToExcel=True)
# Bold Driver
NN3 = NeuralNetwork(0.1, randSeed=1)
NN3.run(epochs=100000, boldDriver=True, showIncrements=False,
        showTable=True, showGraph=False, outputToExcel=True)
# Annealing
NN4 = NeuralNetwork(0.1, randSeed=1)
NN4.run(epochs=100000, annealing=True, showIncrements=False,
        showTable=True, showGraph=False, outputToExcel=True)

# Close excel writer
writer.close()
