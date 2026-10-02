import numpy, numpy.matlib

def circularDistance(arrayX,arrayY,input='deg',output='deg',diffType='pairwise'):
    if ((len(arrayX.shape)>1) and (arrayX.shape[1]>arrayX.shape[0])):
        arrayX = numpy.transpose(arrayX)

    if ((len(arrayY.shape)>1) and (arrayY.shape[1]>arrayY.shape[0])):
        arrayY = numpy.transpose(arrayY)
    
    if ((type(arrayY) is numpy.ndarray) and (arrayX.shape[0] != arrayY.shape[0])):
        raise Exception("circularStats::circularDistance: arrays do not have the same size.")

    if (input=='deg'):
        arrayX = arrayX*numpy.pi/180
        arrayY = arrayY*numpy.pi/180

    distances = None
    if (diffType=='pairwise'):
        distances = numpy.angle(numpy.divide(numpy.exp(1j*arrayX),numpy.exp(1j*arrayY)))
    elif (diffType=='extended'):
        extendedX = numpy.transpose(numpy.matlib.repmat(arrayX,arrayY.shape[0],1))
        extendedY = numpy.matlib.repmat(arrayY,arrayX.shape[0],1)
        distances = numpy.angle(numpy.divide(numpy.exp(1j*extendedX),numpy.exp(1j*extendedY)))
    distances = distances

    if (output=='deg'):
        distances *= 180/numpy.pi
    return distances

def circularMean(inputArray,input='deg',output='deg',weights=None):
    if ((len(inputArray.shape)>1) and (inputArray.shape[1]>inputArray.shape[0])):
        inputArray = numpy.transpose(inputArray)
    
    if (input=='deg'):
        inputArray = inputArray*numpy.pi/180

    if (weights is None):
        weights = numpy.ones(inputArray.shape)
    else:
        if ((len(weights.shape)>1) and (weights.shape[1]>weights.shape[0])):
            weights = numpy.transpose(weights)

    if (type(inputArray) is numpy.ndarray):
        weightedValues = numpy.transpose(weights)@numpy.exp(1j*inputArray)
        mean = numpy.angle(weightedValues)
    else:
        mean = inputArray
    
    if (output=='deg'):
        mean *= 180/numpy.pi
    return mean

def circularMedian(inputArray,input='deg',output='deg'):
    if ((len(inputArray.shape)>1) and (inputArray.shape[1]>inputArray.shape[0])):
        inputArray = numpy.transpose(inputArray)

    if (input=='deg'):
        inputArray = inputArray*numpy.pi/180

    inputArray = inputArray % (numpy.pi*2)

    m1 = numpy.sum(circularDistance(inputArray,inputArray,input=input,output=input,diffType='extended')>0,axis=0)
    m2 = numpy.sum(circularDistance(inputArray,inputArray,input=input,output=input,diffType='extended')<0,axis=0)

    mdiff = numpy.abs(m1-m2)

    m = numpy.min(mdiff)
    index = numpy.argmin(mdiff)
    
    if (m>1):
        raise Exception("circularStats::circularMedian: can not be calculated, multiple candidate values.")
    
    median = circularMean(inputArray[index],input=input,output=input)
    c1 = numpy.abs(circularDistance(circularMean(inputArray,input=input,output=input),median,input=input,output=input))
    c2 = numpy.abs(circularDistance(circularMean(inputArray,input=input,output=input),median+numpy.pi,input=input,output=input))
    if (c1 > c2):
        median = (median+numpy.pi)%(2*numpy.pi)

    if (output=='deg'):
        median *= 180/numpy.pi
    return median

def circularMoment(inputArray,binWeights=None,order=1,centralMoments=False,input='deg'):
    if ((len(inputArray.shape)>1) and (inputArray.shape[1]>inputArray.shape[0])):
        inputArray = numpy.transpose(inputArray)

    if (input=='deg'):
        inputArray = inputArray*numpy.pi/180

    if (centralMoments):
        theta = circularMean(inputArray,input='rad',output='rad',weights=binWeights)
        inputArray = circularDistance(inputArray,theta,input='rad',output='rad')
    
    cbar = numpy.sum(numpy.cos(order*inputArray@numpy.transpose(binWeights)))/inputArray.shape[0]
    sbar = numpy.sum(numpy.sin(order*inputArray@numpy.transpose(binWeights)))/inputArray.shape[0]
    mp = cbar+1j*sbar

    rho = abs(mp)
    mu = numpy.angle(mp)

    return mp,rho,mu

def circularVectorR(inputArray,binWeights=None,binSpacing=0,input='deg'):
    if ((len(inputArray.shape)>1) and (inputArray.shape[1]>inputArray.shape[0])):
        inputArray = numpy.transpose(inputArray)

    if (input=='deg'):
        inputArray = inputArray*numpy.pi/180
        binSpacing = binSpacing*numpy.pi/180
    
    if (binWeights is None):
        binWeights = numpy.ones(inputArray.shape)
    
    distanceR = numpy.transpose(binWeights)@numpy.exp(1j*inputArray)
    distanceR = numpy.abs(distanceR)/numpy.sum(binWeights)

    if (binSpacing!=0):
        correctionFactor = binSpacing/2/numpy.sin(binSpacing/2)
        distanceR = correctionFactor*distanceR

    return distanceR

def circularVariance(inputArray,binWeights=None,binSpacing=0,input='deg'):
    if ((len(inputArray.shape)>1) and (inputArray.shape[1]>inputArray.shape[0])):
        inputArray = numpy.transpose(inputArray)
    
    if (input=='deg'):
        inputArray = inputArray*numpy.pi/180
    
    if (binWeights is None):
        binWeights = numpy.ones(inputArray.shape)
    
    distanceR = circularVectorR(inputArray,binWeights,binSpacing,input='rad')

    return 1-distanceR

def circularSkewness(inputArray,binWeights=None,input='deg'):
    if ((len(inputArray.shape)>1) and (inputArray.shape[1]>inputArray.shape[0])):
        inputArray = numpy.transpose(inputArray)

    if (input=='deg'):
        inputArray = inputArray*numpy.pi/180
    
    if (binWeights is None):
        binWeights = numpy.ones(inputArray.shape)

    R = circularVariance(inputArray,binWeights,input='rad')
    theta = circularMean(inputArray,input='rad',output='rad',weights=binWeights)

    _,rho2,mu2 = circularMoment(inputArray,binWeights,2,True,input='rad')

    b = numpy.transpose(binWeights)@numpy.sin(2*circularDistance(inputArray,theta))/numpy.sum(binWeights)
    b0 = rho2*numpy.sin(circularDistance(mu2,2*theta))/(1-R)**(2/3)

    return b,b0

def rayleighTest(inputArray,binWeights=None,binSpacing=0,input='deg'):
    if ((len(inputArray.shape)>1) and (inputArray.shape[1]>inputArray.shape[0])):
        inputArray = numpy.transpose(inputArray)

    if (input=='deg'):
        inputArray = inputArray*numpy.pi/180
        binSpacing = binSpacing*numpy.pi/180
    
    distanceR = None    
    totalItems = 0

    if (binWeights is None):
        distanceR = circularVectorR(inputArray,input='rad')
        totalItems = len(inputArray)
    else:
        distanceR = circularVectorR(inputArray,binWeights,binSpacing,input='rad')
        totalItems = numpy.sum(binWeights)
    
    rayleighR = totalItems*distanceR
    rayleighZ = (rayleighR**2)/totalItems
    rayleighP = numpy.exp(numpy.sqrt(1+4*totalItems+4*(totalItems**2-rayleighR**2))-(1+2*totalItems))

    return rayleighZ,rayleighP