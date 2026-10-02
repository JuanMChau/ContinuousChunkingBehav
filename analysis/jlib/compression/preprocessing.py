import numpy
from scipy.stats import vonmises
from scipy.optimize import minimize

def _guessingOnCircleModel(angle,kappa,guessing):
    return (1-guessing)*vonmises.pdf(angle,kappa,loc=0)+guessing/(2*numpy.pi)

def _guessingOnCircleNLL(params,data):
    kappa,guessing = params
    if (kappa<0) or not (0<=guessing<=1):
        return numpy.inf
    pdfValues = _guessingOnCircleModel(data,kappa,guessing)
    return -numpy.sum(numpy.log(pdfValues))

def fitGuessingOnCircle(data,initParams,paramBounds):
    return minimize(_guessingOnCircleNLL,initParams,args=(data,),bounds=paramBounds)