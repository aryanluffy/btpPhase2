from cProfile import label
from matplotlib.pyplot import plot
import pandas as pd
from pip import main
from sklearn.model_selection import train_test_split


def calcCorrelation(score,labels):
    meanScore=0
    meanLabel=0
    for i in range(0,len(score)):
        meanScore+=score[i]
        meanLabel+=1-labels[i]
    # meanLabel=0
    meanScore/=len(score)
    meanLabel/=len(score)
    numerator=0
    denominator1=0
    denominator2=0
    for i in range(0,len(score)):
        numerator+=(score[i]-meanScore)*(1-labels[i]-meanLabel)
        denominator1+=(score[i]-meanScore)**2
        denominator2+=(1-labels[i]-meanLabel)**2
    print("Correlation: ",numerator/((denominator1*denominator2)**0.5))
    return numerator/((denominator1*denominator2)**0.5)

def findBestSeparation(scores,labels,topK=10):
    actualList = []
    for i in range(0,len(scores)):
        actualList.append([scores[i],labels[i]])
    correleation=calcCorrelation(scores,labels)
    actualList.sort()
    # print(actualList[1250:])
    # actualList=actualList[-100:-1]
    pos=0
    val=2*actualList[0][1]-1
    curr=val
    labelsSum=labels[0]
    for i in range(1,len(actualList)):
        curr+=2*actualList[i][1]-1
        labelsSum+=actualList[i][1]
        if curr>val:
            val=curr
            pos=i
    inversionsCount=0
    for i in range(0,len(actualList)):
        for j in range(i+1,len(actualList)):
            if actualList[i][1]==0 and actualList[j][1]==1:
                inversionsCount+=1
    print("Inversions Count: ",inversionsCount)
    wrongPredTop=0
    for i in range(len(actualList)-topK,len(actualList)):
        if actualList[i][1]:
            wrongPredTop+=1
    print("top_k_error: ",wrongPredTop/topK)
    wrongPred=0
    for i in range(0,topK):
        if actualList[i][1]==0:
            wrongPred+=1
    print("bottom_k_error: ",wrongPred/topK)
    return [correleation,inversionsCount,wrongPredTop/topK,wrongPred/topK]

def confusionMatrix(predicted,labels):
    matrix = [[0,0],[0,0]]
    correct=0
    for i in range(0,len(predicted)):
        if labels[i]:
            correct+=1
    print("top_k_error: ",correct/len(predicted))
    # print("F1: ",f1)
    return matrix

def mainFunction(powerOfN=1):
    df=pd.read_csv("BtpData/gpt.json")
    # dataSetSize=int(input("Enter training dataset size: "))
    dataSetSize=1381
    # df=df.sample(n=dataSetSize)
    train=df[0:dataSetSize]
    test=df[dataSetSize:]
    # train, test = train_test_split(df, test_size=0.2)
    df=train

    minProb=[]
    perplexity=[]
    sum=[]
    maxMin=[]
    maxMed=[]
    medMin=[]
    med=[]
    labels=[]
    weightedSum=[]
    actualPerplexity=[]
    bleu=[]

    for index in df.index:
        minProb.append(df['min'][index])
        perplexity.append(df['sum'][index])
        sum.append(perplexity[-1])
        perplexity[-1]=sum[-1]/(len(df['Sentence'][index]))**(powerOfN)
        actualPerplexity.append(sum[-1]/(len(df['Sentence'][index])))
        maxMin.append(df['max/min'][index])
        maxMed.append(df['max/med'][index])
        medMin.append(df['med/min'][index])
        med.append(df['med'][index])
        weightedSum.append(df['weightedSum'][index])
        labels.append(df['Label'][index])
        bleu.append(df['bleu'][index])


    # minProb=minProb[0:1000]
    # perplexity=perplexity[0:1000]
    import numpy as np
    # print("max_min_diff")
    labels2=[]
    wrong=0
    correct=0
    for x in labels:
        labels2.append(x)
        if x:
            correct+=1
        else:
            wrong+=1
    print("Number of 1's: ",correct)
    print("Number of 0's: ",wrong)
    # topk=int(input("Enter top k: "))
    topk=50
    # print("minProb")
    # bestSep=findBestSeparation(minProb,labels2,topK=topk)

    # print("\nSum")
    # bestSep=findBestSeparation(sum,labels,topK=topk)

    # print("\nmax/min")
    # bestSep=findBestSeparation(maxMin,labels,topK=topk)

    # print("\nmax/med")
    # bestSep=findBestSeparation(maxMed,labels,topK=topk)

    # print("\nmed/min")
    # bestSep=findBestSeparation(medMin,labels,topK=topk)

    # print("\nmed")
    # bestSep=findBestSeparation(med,labels,topK=topk)

    print("\nPerplexity/length^0.4")
    return findBestSeparation(perplexity,labels,topK=topk)

    # print("\nweighted Sum")
    # bestSep=findBestSeparation(weightedSum,labels,topK=topk)

    # print("\nPerplexity")
    # bestSep=findBestSeparation(actualPerplexity,labels,topK=topk)

    # print("\nBleu")
    # bestSep=findBestSeparation(bleu,labels,topK=topk)

def varyingWeights(alpha=1,topk=50,modelName="gpt"):
    import json

    with open('BtpData/'+modelName+'.json', 'r') as f:
        data = json.load(f)
    
    scores=[]
    labels=[]
    for key in data.keys():
        val=0
        for i in range(0,len(data[key]['contextualProb'])):
            val+=data[key]['contextualProb'][i]*(alpha**i)
        scores.append(val)
        labels.append(int(data[key]['label']))
    return findBestSeparation(scores,labels,topK=topk)

def varyingK(alpha=1,topk=50,modelName="bert"):
    import json

    with open('BtpData/'+modelName+'.json', 'r') as f:
        data = json.load(f)
    
    scores=[]
    labels=[]
    for key in data.keys():
        val=0
        for i in range(0,len(data[key]['contextualProb'])):
            val+=data[key]['contextualProb'][i]/((len(key))**alpha)
        scores.append(val)
        labels.append(int(data[key]['label']))
    return findBestSeparation(scores,labels,topK=topk)

# powers=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
def varyingPowers(topk=50,modelName="bert"):
    powers=[]
    for i in range(0,41):
        powers.append(i/40)
    correlations=[]
    inversions=[]
    topKError=[]
    bottomKError=[]

    for power in powers:
        print("Power: ",power)
        dataPoints=varyingWeights(alpha=power,topk=topk,modelName=modelName)
        inversions.append(dataPoints[1])
        correlations.append(dataPoints[0])
        topKError.append(dataPoints[2])
        bottomKError.append(dataPoints[3])

    # import matplotlib.pyplot as plt

    # plt.plot(powers,correlations)
    # plt.xlabel("Power")
    # plt.ylabel("Correlation")
    # plt.show()

    # mainFunction(powerOfN=0.6)
    # plt.plot(powers,inversions)
    # plt.xlabel('powers')
    # plt.ylabel('Inversions')
    # plt.show()

    # plt.plot(powers,topKError)
    # plt.xlabel('powers')
    # plt.ylabel('topk')
    # plt.show()

    # plt.plot(powers,bottomKError)
    # plt.xlabel('powers')
    # plt.ylabel('bottomK')
    # plt.show()
    return correlations,inversions,topKError,bottomKError,powers

# varyingWeights(alpha=5.0)

def varyingKFor(topk=50,modelName='bert'):
    powers=[]
    for i in range(-40,41):
        powers.append(i/40)
    correlations=[]
    inversions=[]
    topKError=[]
    bottomKError=[]

    for power in powers:
        print("Power: ",power)
        dataPoints=varyingK(alpha=power,topk=topk,modelName=modelName)
        inversions.append(dataPoints[1])
        correlations.append(dataPoints[0])
        topKError.append(dataPoints[2])
        bottomKError.append(dataPoints[3])

    # import matplotlib.pyplot as plt

    # plt.plot(powers,correlations)
    # plt.xlabel("Power")
    # plt.ylabel("Correlation")
    # plt.show()

    # mainFunction(powerOfN=0.6)
    # plt.plot(powers,inversions)
    # plt.xlabel('powers')
    # plt.ylabel('Inversions')
    # plt.show()

    # plt.plot(powers,topKError)
    # plt.xlabel('powers')
    # plt.ylabel('topk')
    # plt.show()

    # plt.plot(powers,bottomKError)
    # plt.xlabel('powers')
    # plt.ylabel('bottomK')
    # plt.show()
    return correlations,inversions,topKError,bottomKError,powers

# varyingKFor(topk=200)

modelNames=['gpt','gpt2Small','gpt2Medium','gpt2Large','gptNeo']


import matplotlib.pyplot as plt

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingPowers(topk=200,modelName=name)
    plt.plot(powers,correlations,label=name)
    plt.xlabel("Base")
    plt.ylabel("Correlation")

plt.legend()
plt.show()
plt.clf()

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingKFor(topk=200,modelName=name)
    plt.plot(powers,correlations,label=name)
    plt.xlabel("Power")
    plt.ylabel("Correlation")

plt.legend()
plt.show()
plt.clf()

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingPowers(topk=200,modelName=name)
    plt.plot(powers,inversions,label=name)
    plt.xlabel("Base")
    plt.ylabel("Inversions")

plt.legend()
plt.show()
plt.clf()

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingKFor(topk=200,modelName=name)
    plt.plot(powers,inversions,label=name)
    plt.xlabel("Power")
    plt.ylabel("Inversions")

plt.legend()
plt.show()
plt.clf()

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingPowers(topk=200,modelName=name)
    plt.plot(powers,topKError,label=name)
    plt.xlabel("Base")
    plt.ylabel("TopK")

plt.legend()
plt.show()
plt.clf()

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingKFor(topk=200,modelName=name)
    plt.plot(powers,topKError,label=name)
    plt.xlabel("Power")
    plt.ylabel("TopK")

plt.legend()
plt.show()
plt.clf()

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingPowers(topk=200,modelName=name)
    plt.plot(powers,bottomKError,label=name)
    plt.xlabel("Base")
    plt.ylabel("BottomK")

plt.legend()
plt.show()
plt.clf()

for name in modelNames:
    correlations,inversions,topKError,bottomKError,powers=varyingKFor(topk=200,modelName=name)
    plt.plot(powers,bottomKError,label=name)
    plt.xlabel("Power")
    plt.ylabel("BottomK")

plt.legend()
plt.show()
plt.clf()



