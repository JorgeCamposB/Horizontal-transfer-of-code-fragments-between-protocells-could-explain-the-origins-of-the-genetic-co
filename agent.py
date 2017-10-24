import random
import math

#The class for the agent

class Agent:
    def __init__(self,numInputs,numHidden,numOutputs,
                 initWeightSD,learningRate):
        """ Each agent is defined as a primitive translation mechanism represented by a 
            feed-forward neural network
            
            Parameters to construct each agent:
                numInputs    -----> Number neurons needed for the coded inputs (Signal -----> Codon)
                numHidden    -----> Number of hidden neurons
                numOutputs   -----> Number of neurons needed for the coded outputs (Meaning ------> Aminoacid)
                initWeightSD -----> Standard deviation of the initial weights of the neural network with mean at 0
                learningRate -----> Learning rate of the neural network
            
            Extra parameters used on each agent:
                firstTimeLearner -----> A boolean that represent if the agent doesn't receive any transfer during a simulation
                noOfLE           -----> Number of learning episodes where the agent performed as learner
        """
        #We add 1 to the number of inputs for the bias node as Brace et. al. 2015). 
        #The bias node works like the bias for each node separately,
        #that only depends on the weight from the bias node (that always has a value of 1) 
        #to the neuron that  should be affected by that bias.

        numInputs += 1

        self.inputToHidden = [] 
        self.hiddenToOutput = []
        self.numInputs = numInputs 
        self.numHidden = numHidden
        self.numOutputs = numOutputs
        self.learningRate = learningRate
        self.firstTimeLearner = True
        self.noOfLE = 0

        for i in range(numInputs):
            workingList = []
            for h in range(numHidden):
                newWeight = random.normalvariate(0.0, initWeightSD)
                workingList.append(newWeight)
            self.inputToHidden.append(workingList)


        for h in range(numHidden + 1):     ## +1 to cover the bias-to-output connections
            workingList = []
            for o in range(numOutputs):
                newWeight = random.normalvariate(0.0, initWeightSD)
                workingList.append(newWeight)
            self.hiddenToOutput.append(workingList)



    # A function to print the values of the weights of the network if needed

    def printNetwork(self, name):
        print (name, "network weights")
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                print (self.inputToHidden[i][j],',',end='')
            print()
        print ("---------------------------------------")
        for j in range(self.numHidden):
            for k in range(self.numOutputs):
                print (self.hiddenToOutput[j][k],',',end='')
            print()
    


    # A function to calculate the output of the neural network
    
    def calcNetOutput(self,inputToNet,wantHiddenLevels):
        """ Inputs of this function is listed as follows:
                inputToNet       ------> Input is the list of real-valued inputs (Signal -----> Codon).  
                wantHiddenLevels ------> A Boolean that specify whether you'd like the hidden layer 
                                         activation levels returned or not.
                                         
            Output is a list of real-valued outputs, and optionally with the hidden layer
            activation levels.  
                *Both are sigmoid-functioned before being returned.
        """

        ## immediately add 1.0 to the input list: this is the bias node
        inputWithBias = inputToNet[:]
        inputWithBias.append(1.0)

        hiddenActivationLevels = [ 0.0 ] * self.numHidden
        outputActivationLevels = [ 0.0 ] * self.numOutputs

        for i in range(self.numInputs):
            for h in range(self.numHidden):
                hiddenActivationLevels[h] += ( inputWithBias[i]
                                               * self.inputToHidden[i][h] )

        for h in range(self.numHidden):
            hiddenActivationLevels[h] = self.sigmoid(hiddenActivationLevels[h])

        hiddenActivationLevels.append(1.0)  ## this is the bias-to-output node
                
        for h in range(self.numHidden + 1):    ## +1 to cover the bias-to-output connections
            for o in range(self.numOutputs):
                outputActivationLevels[o] += ( hiddenActivationLevels[h]
                                               * self.hiddenToOutput[h][o] )

        ## Note that we sigmoid the output functions before we send them back
        for o in range(self.numOutputs):
            outputActivationLevels[o] = self.sigmoid(outputActivationLevels[o])

        if wantHiddenLevels:
            return outputActivationLevels, hiddenActivationLevels
        else:
            return outputActivationLevels

    #The sigmoid function

    def sigmoid(self,x):
        """ Inputs: x ---> Value to apply
            Output: The sigmoid of the 'x' value.
                So the return value is bounded by 0 and 1, but the input value
                can be anything (pos or neg, bounded by infinity)."""
        retVal = 1.0 / (1.0 + math.exp(0.0 - x) )
        return retVal
        
    #A function for the training episode (A reception of an amino acid)        
    
    def trainingEpisode(self,targetOutput,actualOutput,
                        hiddenOutput,actualInput):
        """ Inputs:
                targetOutput  -----> List of desired outputs of the agent (The aminoacid 
                                     that should be coded like the speaker)
                actualOutput  -----> The current list of outputs of the agent
                hiddenOutput  -----> The list of outputs of the hidden layer (The hidden
                                     levels of the function calcNetOutput)
                actualInput   -----> The list of the current inputs to the agent (Codon)
            Output:
                List of errors per output neuron                
        
        The backpropagation process is coded here.
        This process happens in each transfer a certain number of epochs.
        
        We do one training event where the weights of the network
        get properly updated for a given target output and a given
        actual output.  ."""

        # as before, add 1.0 to the input list to implement the bias node
        inputWithBias = actualInput[:]
        inputWithBias.append(1.0)

        hiddenWithBias = hiddenOutput[:]
        hiddenWithBias.append(1.0)
        
        # calculate deltaKs: basically an error measure per output neuron
        deltaK = [ (t - y) * y * (1 - y)
                   for t, y in zip(targetOutput, actualOutput) ]

        # train the hidden-to-output connections
        for j in range(self.numHidden):   ## +1 for the bias-to-output connections
            for k in range(self.numOutputs):
                deltaW = self.learningRate * deltaK[k] * hiddenWithBias[j]
                self.hiddenToOutput[j][k] += deltaW

        # calculate deltaJs: basically an error measure for hidden neurons
        deltaJ = [ 0.0 ] * self.numHidden
        for j in range(self.numHidden):
            sigma = 0.0
            for k in range(self.numOutputs):
                sigma += self.hiddenToOutput[j][k] * deltaK[k]
            deltaJ[j] = hiddenOutput[j] * ( 1 - hiddenOutput[j] ) * sigma

        # train the input-to-hidden connections
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                deltaW = self.learningRate * deltaJ[j] * inputWithBias[i]
                self.inputToHidden[i][j] += deltaW 
        return deltaK