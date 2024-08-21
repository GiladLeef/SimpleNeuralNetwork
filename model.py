import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return x * (1 - x)

def mseLoss(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)

class NeuralNetwork:
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.weightsInputHidden = np.random.randn(inputSize, hiddenSize) * np.sqrt(2. / inputSize)
        self.biasHidden = np.zeros(hiddenSize)
        self.weightsHiddenOutput = np.random.randn(hiddenSize, outputSize) * np.sqrt(2. / hiddenSize)
        self.biasOutput = np.zeros(outputSize)
    
    def feedforward(self, X):
        self.hiddenInput = np.dot(X, self.weightsInputHidden) + self.biasHidden
        self.hiddenOutput = sigmoid(self.hiddenInput)
        self.outputInput = np.dot(self.hiddenOutput, self.weightsHiddenOutput) + self.biasOutput
        self.output = sigmoid(self.outputInput)
        return self.output
    
    def backpropagation(self, X, y, learningRate):
        outputError = y - self.output
        outputDelta = outputError * sigmoidDerivative(self.output)
        hiddenError = np.dot(outputDelta, self.weightsHiddenOutput.T)
        hiddenDelta = hiddenError * sigmoidDerivative(self.hiddenOutput)
        self.weightsHiddenOutput += np.dot(self.hiddenOutput.T, outputDelta) * learningRate
        self.biasOutput += np.sum(outputDelta, axis=0) * learningRate
        self.weightsInputHidden += np.dot(X.T, hiddenDelta) * learningRate
        self.biasHidden += np.sum(hiddenDelta, axis=0) * learningRate
    
    def train(self, X, y, epochs, learningRate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learningRate)
            if epoch % 1000 == 0:
                loss = mseLoss(y, self.output)
                print(f'Epoch {epoch}, Loss: {loss}')

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(inputSize=2, hiddenSize=4, outputSize=1)
    nn.train(X, y, epochs=1000, learningRate=1)
    
    print("Predictions after training:")
    print(nn.feedforward(X))
