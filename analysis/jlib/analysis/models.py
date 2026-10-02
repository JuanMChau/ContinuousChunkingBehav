from jlib.compression import constants
from jlib.analysis import models
from joblib import Parallel, delayed
import numpy
from scipy.stats import vonmises
from scipy.optimize import least_squares
from scipy.optimize import minimize
import sklearn

def _fastA1inv(R):
    # Create an array of NaNs with the same shape as R
    R = numpy.asarray(R)
    K = numpy.full(R.shape,numpy.nan)

    # Conditions
    ix1 = (R >= 0) & (R < 0.53)
    ix2 = (R >= 0.53) & (R < 0.85)
    ix3 = (R >= 0.85) & (R <= 1)

    # Apply formulas
    K[ix1] = 2 * R[ix1] + R[ix1]**3 + (5 * R[ix1]**5) / 6
    K[ix2] = -0.4 + 1.39 * R[ix2] + 0.43 / (1 - R[ix2])
    K[ix3] = 1 / (R[ix3]**3 - 4 * R[ix3]**2 + 3 * R[ix3])

    return K

def _fixKappa(kappa,numPoints):
    if numPoints<15:
        if kappa<2:
            kappa = max((kappa-2)/(numPoints*kappa),0)
        else:
            kappa *= (numPoints-1)**3/(numPoints**3+numPoints)
    
    return kappa

def _fitMisbindingMultiVonMises(data,targets,nonTargets,startingParams,
                                separateKappas,maxIterations=10000,maxDLogLikelihood=1e-4):
    error = (data-targets+numpy.pi)%(2*numpy.pi)-numpy.pi
    nonError = (data[:, None]-nonTargets+numpy.pi)%(2*numpy.pi)-numpy.pi

    logLikelihood = 0
    dLogLikelihood = 0
    iterations = 0

    if separateKappas:
        kappaT,kappaNT,targetP,nonTargetP,guessingP = startingParams
    else:
        kappaT,targetP,nonTargetP,guessingP = startingParams
        kappaNT = 0

    while(1):
        targetW = targetP*vonmises.pdf(error,kappaT,loc=0)
        guessingW = numpy.ones(data.shape)*guessingP/(2*numpy.pi)

        if (nonTargets.shape[1]==0):
            nonTargetW = numpy.zeros(nonError.shape)
        else:
            if separateKappas:
                nonTargetW = nonTargetP/nonTargets.shape[1]*vonmises.pdf(nonError,kappaNT,loc=0)
            else:
                nonTargetW = nonTargetP/nonTargets.shape[1]*vonmises.pdf(nonError,kappaT,loc=0)
        
        weights = numpy.sum(numpy.column_stack((targetW,nonTargetW,guessingW)),axis=1)

        dLogLikelihood = logLikelihood-numpy.sum(numpy.log(weights))
        logLikelihood = numpy.sum(numpy.log(weights))

        if (abs(dLogLikelihood)<maxDLogLikelihood) or (iterations>=maxIterations):
            break

        targetP = numpy.sum(targetW/weights)/data.shape[0]
        nonTargetP = numpy.sum(numpy.sum(nonTargetW,axis=1)/weights)/data.shape[0]
        guessingP = numpy.sum(guessingW/weights)/data.shape[0]

        if separateKappas:
            ST = numpy.sum(numpy.sin(error)*targetW/weights)
            CT = numpy.sum(numpy.cos(error)*targetW/weights)
            
            if (numpy.sum(targetW/weights)==0):
                kappaT = 0
            else:
                RT = numpy.sqrt(ST**2+CT**2)/numpy.sum(targetW/weights)
                kappaT = _fastA1inv(RT)
                kappaT = _fixKappa(kappaT,data.shape[0])
            
            if (nonTargets.shape[1]>0):
                SNT = numpy.sum(numpy.sin(nonError)*nonTargetW/weights[:, None])
                CNT = numpy.sum(numpy.cos(nonError)*nonTargetW/weights[:, None])

                if (numpy.sum(nonTargetW/weights[:,None])==0):
                    kappaNT = 0
                else:
                    RNT = numpy.sqrt(SNT**2+CNT**2)/numpy.sum(nonTargetW/weights[:, None])
                    kappaNT = _fastA1inv(RNT)
                    kappaNT = _fixKappa(kappaNT,data.shape[0])
            else:
                kappaNT = 0
        else:
            relativeWeight = numpy.column_stack((targetW/weights,nonTargetW/weights[:,None]))

            S = numpy.column_stack((numpy.sin(error),numpy.sin(nonError)))
            C = numpy.column_stack((numpy.cos(error),numpy.cos(nonError)))
            r = numpy.column_stack((numpy.sum(numpy.sum(S*relativeWeight)),numpy.sum(numpy.sum(C*relativeWeight))))

            if (numpy.sum(numpy.sum(relativeWeight))==0):
                kappaT = 0
            else:
                R = numpy.sqrt(numpy.sum(r*r))/numpy.sum(numpy.sum(relativeWeight))
                kappaT = _fastA1inv(R)
                kappaT = _fixKappa(kappaT,data.shape[0])

        iterations += 1

    if (iterations>=maxIterations):
        modelEstimation = {'kappaT':numpy.nan,'kappaNT':numpy.nan,'targetP':numpy.nan,
                           'nonTargetP':numpy.nan,'guessingP':numpy.nan}
        logLikelihood = numpy.nan
        weights = numpy.nan
    else:
        modelEstimation = {'kappaT':kappaT,'kappaNT':kappaNT,'targetP':targetP,
                           'nonTargetP':nonTargetP,'guessingP':guessingP}
        weights = numpy.column_stack((targetW,numpy.sum(nonTargetW,axis=1),guessingW))/(targetW+numpy.sum(nonTargetW,axis=1)+guessingW)[:,None]

    return modelEstimation,logLikelihood,weights

def fitMisbindingMultiVonMises(data,targets,nonTargets=None,separateKappas=False):
    startingKappas = [1,10,100]
    startingMisbindings = [0.01,0.1,0.4]
    startingGuessing = [0.01,0.1,0.4]

    if (nonTargets is None):
        nonTargets = numpy.zeros((data.shape[0],0))
        startingMisbindings = [0]
    
    outputLogLikelihood = -numpy.inf
    outputParams = [None,None,None,None]
    outputPosteriors = None

    if separateKappas:
        for kappaT in startingKappas:
            for kappaNT in startingKappas:
                for misbinding in startingMisbindings:
                    for guessing in startingGuessing:
                        params,logLikelihood,posteriors = _fitMisbindingMultiVonMises(data,targets,nonTargets,
                                                                                      [kappaT,kappaNT,1-guessing-misbinding,misbinding,guessing],
                                                                                      separateKappas)
                        if (logLikelihood>outputLogLikelihood):
                            outputLogLikelihood = logLikelihood
                            outputParams = params
                            outputPosteriors = posteriors
    else:
        for kappa in startingKappas:
            for misbinding in startingMisbindings:
                for guessing in startingGuessing:
                    params,logLikelihood,posteriors = _fitMisbindingMultiVonMises(data,targets,nonTargets,
                                                                                  [kappa,1-guessing-misbinding,misbinding,guessing],
                                                                                  separateKappas)
                    if (logLikelihood>outputLogLikelihood):
                        outputLogLikelihood = logLikelihood
                        outputParams = params
                        outputPosteriors = posteriors
        
    return outputParams,outputLogLikelihood,outputPosteriors

def evaluateMisbindingMultiVonMises(data,params,targets,nonTargets=None,separateKappas=False):
    error = (data-targets+numpy.pi)%(2*numpy.pi)-numpy.pi
    pdfValues = params['targetP']*vonmises.pdf(error,params['kappaT'],loc=0)+params['guessingP']/(2*numpy.pi)

    if (nonTargets is not None):
        nonError = (data[:, None]-nonTargets+numpy.pi)%(2*numpy.pi)-numpy.pi
        if (separateKappas):
            pdfValues = pdfValues+numpy.sum(params['nonTargetP']/nonTargets.shape[1]*vonmises.pdf(nonError,params['kappaNT'],loc=0),axis=1)
        else:
            pdfValues = pdfValues+numpy.sum(params['nonTargetP']/nonTargets.shape[1]*vonmises.pdf(nonError,params['kappaT'],loc=0),axis=1)
            
    logLikelihood = numpy.sum(numpy.log(pdfValues))

    return pdfValues,logLikelihood


# def _sigmoid(x,b,L,k,x0):
#      y = b+L/(1+numpy.exp(-k*(x-x0)))
#      return y

def _sigmoid_optimization(params,x,y):
    return params[0]+params[1]/(1+numpy.exp(-params[2]*(x-params[3])))-y

def fitSigmoid(dataX,dataY,startingParams,paramBounds):
    return least_squares(_sigmoid_optimization,x0=startingParams,bounds=paramBounds,args=(dataX,dataY))

def _line_optimization(params,x,y):
    return params[0]*x+params[1]-y

def fitLine(dataX,dataY,startingParams,paramBounds):
    return least_squares(_line_optimization,x0=startingParams,bounds=paramBounds,args=(dataX,dataY))

def _biasedModel(angle,location,kappa,bias,guessing):
    return (1-guessing)*vonmises.pdf(angle,kappa,loc=bias*location)+guessing/(2*numpy.pi)

def _biasedNLL(params,data,location):
    kappa,bias,guessing = params
    if (kappa<0) or not (0<=guessing<=1) or not (-1<=bias<=1):
        return numpy.inf
    pdfValues = _biasedModel(data,location,kappa,bias,guessing)
    return -numpy.sum(numpy.log(pdfValues))

def fitBiasedModel(data,location,initParams,paramBounds):
    return minimize(_biasedNLL,initParams,args=(data,location,),bounds=paramBounds,tol=1e-4)

def calculateMetric(LL,nParams,nTrials,metric='LL'):
    if (metric=='LL'):
        return LL
    elif (metric=='AIC'):
        return -2*numpy.array(LL)+2*nParams
    elif (metric=='BIC'):
        return -2*numpy.array(LL)+nParams*numpy.log(numpy.array(nTrials))
    elif (metric=='NormLL'):
        return numpy.array(LL)/numpy.array(nTrials)
    
    return None

def _fitBiasModelsToParticipantCV(data,modelsToTrain,metricToReturn,nFolds,participant):
    participantData = data[data['Subject']==participant]

    participantFits = {model:[] for model in modelsToTrain}
    participantMetrics = {model:[] for model in modelsToTrain}

    allErrors = participantData['Response error'].to_numpy()*numpy.pi/180
    allMeanErrors = participantData['Continuous target'].to_numpy()-participantData['Mean target'].to_numpy()
    allMeanErrors = allMeanErrors*2*numpy.pi/constants.EXP4_RESPONSE_SECTIONS

    trainIndices = []
    testIndices = []

    if (nFolds>1):
        participantKF = sklearn.model_selection.KFold(n_splits=nFolds,shuffle=True,random_state=participant)
        for trainIndex,testIndex in participantKF.split(allErrors):
            trainIndices.append(trainIndex)
            testIndices.append(testIndex)
    else:
        trainIndices.append(numpy.arange(len(allErrors)).astype(int))

    for index in range(len(trainIndices)):
        # Training values
        errorsTrain = allErrors[trainIndices[index]]
        targetsTrain = numpy.zeros(errorsTrain.shape)
        meanErrorsTrain = allMeanErrors[trainIndices[index]]
        # Testing values
        if (len(testIndices)>0):
            errorsTest = allErrors[testIndices[index]]
            targetsTest = numpy.zeros(errorsTest.shape)
            meanErrorsTest = allMeanErrors[testIndices[index]]
        # Misbinding with single, shared kappa
        if ('MBSK' in modelsToTrain):
            params,logLikelihoodTrain,_ = models.fitMisbindingMultiVonMises(errorsTrain,targetsTrain,
                                                                            meanErrorsTrain[:,None])
            participantFits['MBSK'].append(params)
            if (len(testIndices)>0):
                _,logLikelihoodTest = models.evaluateMisbindingMultiVonMises(errorsTest,params,targetsTest,
                                                                             meanErrorsTest[:,None])
                participantMetrics['MBSK'].append(calculateMetric([logLikelihoodTrain,logLikelihoodTest],3,
                                                                  [errorsTrain.shape[0],errorsTest.shape[0]],metricToReturn))
            else:
                participantMetrics['MBSK'].append(calculateMetric([logLikelihoodTrain],3,
                                                                  [errorsTrain.shape[0]],metricToReturn))
        # Misbinding with separate kappas
        if ('MBDK' in modelsToTrain):
            params,logLikelihoodTrain,_ = models.fitMisbindingMultiVonMises(errorsTrain,targetsTrain,
                                                                            meanErrorsTrain[:,None],separateKappas=True)
            participantFits['MBDK'].append(params)
            if (len(testIndices)>0):
                _,logLikelihoodTest = models.evaluateMisbindingMultiVonMises(errorsTest,params,targetsTest,
                                                                             meanErrorsTest[:,None],separateKappas=True)
                participantMetrics['MBDK'].append(calculateMetric([logLikelihoodTrain,logLikelihoodTest],4,
                                                                  [errorsTrain.shape[0],errorsTest.shape[0]],metricToReturn))
            else:
                participantMetrics['MBDK'].append(calculateMetric([logLikelihoodTrain],4,
                                                                  [errorsTrain.shape[0]],metricToReturn))
        # No misbinding
        if ('NMB' in modelsToTrain):
            params,logLikelihoodTrain,_ = models.fitMisbindingMultiVonMises(errorsTrain,targetsTrain)
            participantFits['NMB'].append(params)
            if (len(testIndices)>0):
                _,logLikelihoodTest = models.evaluateMisbindingMultiVonMises(errorsTest,params,targetsTest)
                participantMetrics['NMB'].append(calculateMetric([logLikelihoodTrain,logLikelihoodTest],2,
                                                                 [errorsTrain.shape[0],errorsTest.shape[0]],metricToReturn))
            else:
                participantMetrics['NMB'].append(calculateMetric([logLikelihoodTrain],2,
                                                                 [errorsTrain.shape[0]],metricToReturn))
        # Bias
        if ('Bias' in modelsToTrain):
            params = models.fitBiasedModel(errorsTrain,-meanErrorsTrain,[1,0.5,0.1],
                                                        ((1e-6,numpy.inf),(-1,1),(0,1)))
            logLikelihoodTrain = models._biasedNLL(params.x,errorsTrain,-meanErrorsTrain)
            if (len(testIndices)>0):
                logLikelihoodTest = models._biasedNLL(params.x,errorsTest,-meanErrorsTest)
                participantMetrics['Bias'].append(calculateMetric([-logLikelihoodTrain,-logLikelihoodTest],3,
                                                                  [errorsTrain.shape[0],errorsTest.shape[0]],metricToReturn))
            else:
                participantMetrics['Bias'].append(calculateMetric([-logLikelihoodTrain],3,
                                                                  [errorsTrain.shape[0]],metricToReturn))
            params = {'kappa':params.x[0],'bias':params.x[1],'guessingP':params.x[2]}
            participantFits['Bias'].append(params)
            
    return participant,participantFits,participantMetrics

def fitAllBiasModelsCV(data,modelsToTrain,metricsToReturn,nFolds=10,nJobs=-1):
    if (modelsToTrain=='all'):
        modelsToTrain = ['MBSK','MBDK','NMB','Bias']

    participantList = numpy.unique(data['Subject'])
    outputs = Parallel(n_jobs=nJobs,verbose=10)(delayed(_fitBiasModelsToParticipantCV)(data,modelsToTrain,metricsToReturn,nFolds,
                                                                                       participant) for participant in participantList)
    
    allParticipantFits = {model:{} for model in modelsToTrain}
    allParticipantMetrics = {model:{} for model in modelsToTrain}

    for participant,participantFits,participantMetrics in outputs:
        for fitType in participantFits:
            allParticipantFits[fitType][participant] = participantFits[fitType]
            allParticipantMetrics[fitType][participant] = participantMetrics[fitType]
        
    return allParticipantFits,allParticipantMetrics