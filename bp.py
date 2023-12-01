from NeuralNetwork import *
from NN_Settings import *
import time
import copy

startTime = time.perf_counter()

iniNN = copy.deepcopy(NN)

for i in range(1):  # Replace with your range if needed
    for j in range(1):  # Replace with your range if needed
        # Optional cooling off period for the processor
        if False:
            time.sleep(5)
            NN.hiddenLayerSizes[0] = i
            NN.hiddenLayerSizes[1] = j

        includeTraining = True
        includeValidate = True

        if includeTraining:
            costs, results, percsError = NN.trainNetwork(NN.epochs, './input/ring-separable.txt', False, useLastTrainResults=False)
            NN.plotTraining(costs, percsError, True, True)  # showPlot, savePlot

        if includeValidate:
            print("-----------------------------------")
            print("Now validating")
            results = NN.validateLearning('./input/ring-test.txt')
            NN.plotValidate(results, True, True)  # showPlot, savePlot

        NN = copy.deepcopy(iniNN)
        print("-----------------------------------")
        print("cicle %dx%d ending time: %f" % (i, j, (time.perf_counter() - startTime)))
        print("-----------------------------------")

endTime = time.perf_counter()
print("-----------------------------------")
print("Total Time: " + str(endTime - startTime) + "\n======================")
