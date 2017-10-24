
from Osim import Sim
import numpy as np
import sys

##A function to define initial parameters of the simulation

def init_params(sHL=6,lR=0.1,bS=10,ep=100,trans=30,rdS=500,popS = 10):
    """Inputs are the following parameters for the simulation:
            sHL   ----> Size of the hidden layer of the neural network
            lR    ----> Learning rate of the agents present in the population
            bS    ----> BottleNeck size
            ep    ----> Epochs
            trans ----> Total transfers
            rdS   ----> Random seed
            popS  ----> Population size
        Output is the dictionary that contais all the parameters that a simulation needs.
    """        
    params = {}

    #Agents parameters
    #Here we fixed the number of the input/output neurons to 3/11 to match with
    #each base of the codon for the input and each feature of the aminoacid for
    #the output.

    params['lengthOfInputStrings'] = 3
    params['lengthOfOutputStrings'] = 11 
    params['numInputNeurons'] = params['lengthOfInputStrings']
    params['numOutputNeurons'] = params['lengthOfOutputStrings'] 
    params['numHiddenNeurons'] = sHL
    
    #The maximum input number of inputs and outputs:
    #      64 possible combinations of the four bases
    #      20 possible aminoacids
    params['maxInputStringIndex'] = int(64)
    params['maxOutputStringIndex'] = int(20)
    
    #Simulation parameters
    #       Where the bottleneck is the number of pairs of aminoacid and codon that
    #       will be transferred to the recipient protocell
    params['randomSeed']=rdS
    params['maxTransmissions'] = trans
    params['popSize']= popS
    params['bottleNeck'] = bS
    params['numberOfEpochs'] = ep
    
    #Agent parameters
    params['initWeightSD'] = 0.1
    params['learningRate'] = lR
    return params

    
## Two functions to run a single simulation or multiple simulation with a 
## Parameter sweep
    

def multipleRuns():
    Data = []
    for rdS in [10,30,80,777,555]:
        for sHL in range(4,13):
            for lR in np.arange(0.1,0.6,0.1):
                for bS in range(10,22,2):
                    for ep in range(50,550,50):
                        for trans in range(50,550,50):
                            print("Simulation:",sHL,lR,bS,ep,trans,rdS)
                            params = init_params(sHL,lR,bS,ep,trans,rdS)
                            go = Sim(params)
                            expresivity, compositionality, stab= go.run()
                            Data.append([rdS,sHL,lR,bS,ep,trans,expresivity,compositionality,stab])
                            np.savetxt('Data.csv',Data,delimiter=',')
def singleRun():
    Data = []
    sHL=6
    lR=0.1
    bS=20
    ep=300
    trans=2000
    rdS=1300
    if len(sys.argv)>1:
        rdS=sys.argv[1]
    popS=16 
    print("Simulation:",sHL,lR,bS,ep,trans,rdS,popS)
    params = init_params(sHL,lR,bS,ep,trans,rdS,popS)
    go = Sim(params,reloadFlag = False)
    expresivity, compositionality, stab= go.run(reloadFlag = False)
    Data.append([rdS,sHL,lR,bS,ep,trans,expresivity,compositionality,stab])
    np.savetxt('Data.csv',Data,delimiter=',',fmt='%.4f')                          

singleRun()
