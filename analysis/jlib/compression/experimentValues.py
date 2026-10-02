from jlib.compression import constants
from pandas import isnull, isna
import numpy

def _findReportedElement(element,elementList):
    '''
    Function to get a color position from the participant reported values.
    '''
    return -1 if isnull(element) else elementList.index(element)

def calculateAwareness(subject,elementList=constants.EXP1_COLOR_NAME,pairPrefix='P',
                       nPairs=4,considerProbability=True,forcedPairProbability=0.8):
    '''
    Function to calculate awareness based on participant-reported pairs and percentages.
    '''
    awareness = 0
    for pair in range(nPairs):
        reportedItem1 = _findReportedElement(subject[pairPrefix+str(pair+1)+'1'],elementList)
        reportedItem2 = _findReportedElement(subject[pairPrefix+str(pair+1)+'2'],elementList)

        if((reportedItem1==-1) or (reportedItem2==-1)):
            continue

        index1 = subject['ItemList'].index(reportedItem1)
        index2 = subject['ItemList'].index(reportedItem2)
        if ((index1%2==0) and (index1+1==index2)) or ((index2%2==0) and (index2+1==index1)):
            if (considerProbability):
                reportedPercent = subject['Percent'+str(pair+1)]
                awareness += 1-abs(reportedPercent-forcedPairProbability)/forcedPairProbability
            else:
                awareness += 1
            
    return awareness/nPairs

def calculateBindingErrors(trial,items,expType='Exp1'):
    if (expType not in ['Exp1']):
        return None
    
    if ((trial['key_ans.keys'] is None) or isna(trial['key_ans.keys'])):
        return 0
    
    neighborPos = int(trial['target'])+(1 if trial['target']%2==0 else -1)
    neighborKey = constants.EXP1_KEYS[items[neighborPos]]
    return (1 if trial['key_ans.keys']==neighborKey else 0)

def calculateBindingAndPostPerceptualInference(trial,pairs,items,expType='Exp1'):
    if (expType not in ['Exp1']):
        return None
    
    if ((trial['key_ans.keys'] is None) or isna(trial['key_ans.keys'])):
        return 0
    
    neighborPos = int(trial['target'])+(1 if trial['target']%2==0 else -1)
    neighborPairIndex = pairs.index(items[neighborPos])
    neighborPairIndex += (1 if neighborPairIndex%2==0 else -1)
    neighborPairKey = constants.EXP1_KEYS[pairs[neighborPairIndex]]
    return (1 if trial['key_ans.keys']==neighborPairKey else 0)

def calculateClosestContinuousPrototype(trial,responseAngleOffset,nItems,nResponseSections,expType='Exp3'):
    if (expType not in ['Exp3','Exp4','Exp6']):
        return None
    
    minVal = -90+1+responseAngleOffset*(nResponseSections/360)
    maxVal = 90-responseAngleOffset*(nResponseSections/360)
    
    ranges = numpy.array(list(range(nItems+1)))*nResponseSections/nItems+numpy.array([[minVal],[maxVal]])
    middlePoints = numpy.mean(ranges,axis=0,dtype=int)

    distances = numpy.abs(middlePoints-trial['Continuous target'])
    minPos = numpy.argmin(distances)
    return middlePoints[minPos]

def calculateKValue(accuracy,totalItems=8):
    '''
    Function to calculate effective number of items that were remembered based on participant accuracy.
    '''
    return (accuracy*totalItems*totalItems-totalItems)/(totalItems-1)

def calculatePairAwareness(subject,elementList=constants.EXP1_COLOR_NAME,pairPrefix='P',
                           nPairs=4,considerProbability=True,forcedPairProbability=0.8):
    '''
    Function to calculate awareness for a single pair based on participant-reported information.
    '''
    awareness = [0,0,0,0]
    for pair in range(nPairs):
        reportedItem1 = _findReportedElement(subject[pairPrefix+str(pair+1)+'1'],elementList)
        reportedItem2 = _findReportedElement(subject[pairPrefix+str(pair+1)+'2'],elementList)

        if ((reportedItem1==-1) or (reportedItem2==-1)):
            continue

        index1 = subject['ItemList'].index(reportedItem1)
        index2 = subject['ItemList'].index(reportedItem2)
        if ((index1%2==0) and (index1+1==index2)) or ((index2%2==0) and (index2+1==index1)):
            pairIndex = int(index1/2) if index1<index2 else int(index2/2)
            if (considerProbability):
                reportedPercent = subject['Percent'+str(pair+1)]
                awareness[pairIndex] = 1-abs(reportedPercent-forcedPairProbability)/forcedPairProbability
            else:
                awareness[pairIndex] = 1

    return awareness

def calculatePostPerceptualInference(trial,pairs,expType='Exp1'):
    if (expType not in ['Exp1']):
        return None
    
    if ((trial['key_ans.keys'] is None) or isna(trial['key_ans.keys'])):
        return 0
    
    pairIndex = pairs.index(trial['key'])
    pairIndex += (1 if pairIndex%2==0 else -1)
    pairKey = constants.EXP1_KEYS[pairs[pairIndex]]
    return (1 if trial['key_ans.keys']==pairKey else 0)

def flipContinuousError(trial,nResponseSections,expType='Exp3'):
    if (expType not in ['Exp3','Exp4','Exp6']):
        return None
    
    if (trial['Response error'] is None):
        return None
    
    offsetToMean = (trial['Mean target']-trial['Continuous target'])*360/nResponseSections*numpy.pi/180
    offsetToMean = numpy.angle(numpy.exp(1j*offsetToMean))*180/numpy.pi
    return (trial['Response error'] if offsetToMean>0 else -trial['Response error'])

def getCuedTarget(trial,expType='Exp6'):
    if (expType not in ['Exp6']):
        return None
    
    if (trial['Cue type']=='valid'):
        return trial['Target']
    
    cueChangeMap = {'invalidC':{0:1,1:0,2:3,3:2},
                    'invalid1':{0:2,1:2,2:0,3:0},
                    'invalid2':{0:3,1:3,2:1,3:1}}
    cueChangeMapOrientation = {2:0,0:1,3:2,1:3}
    
    if (trial['Orientation']=='left'):
        return cueChangeMap[trial['Cue type']][trial['Target']]
    else:
        return cueChangeMapOrientation[cueChangeMap[trial['Cue type']][trial['Target']]]

def getElements(elementList,baseElementList):
    '''
    Function to get a color positions from the original experiment ordering.
    '''
    return [baseElementList.index(item) for item in elementList]

def getTrueCuePosition(trial,expType='Exp6'):
    if (expType not in ['Exp6']):
        return None
    
    return constants.EXP6_CUE_POSITIONS[trial['Orientation']][trial['Cued target']]

def getTrueDistanceToMean(trial,nWheelSlices,expType='Exp3'):
    if (expType not in ['Exp3','Exp4','Exp6']):
        return None
    
    distanceToMean = (trial['Continuous target']-trial['Mean target'])*360/nWheelSlices
    return distanceToMean

def getPairPosition(trial,expType='Exp1'):
    if (expType not in ['Exp1']):
        return None
    
    if (trial['target'] in [6,7]):
        return 'Left vertical'
    elif (trial['target'] in [2,3]):
        return 'Right vertical'
    elif (trial['target'] in [0,1]):
        return 'Top horizontal'
    return 'Bottom horizontal'

def selectInvalidCuePosition(trial,expType='Exp6'):
    if (expType not in ['Exp6']):
        return None

    if (trial['Cue type']=='valid'):
        return 'Valid'
    elif (trial['Cue type']=='invalidC'):
        return 'Chunk neighbour'
    elif (((trial['Cue type']=='invalid1') and (trial['Target'] in [0,2])) or
          ((trial['Cue type']=='invalid2') and (trial['Target'] in [1,3]))):
        return 'Other neighbour'
    elif (((trial['Cue type']=='invalid2') and (trial['Target'] in [0,2])) or
          ((trial['Cue type']=='invalid1') and (trial['Target'] in [1,3]))):
        return 'Opposite'
    
    return None

def splitByDistanceToMean(trial,splitSections,unusedAngle,nTargets,nWheelSlices,expType='Exp3'):
    if (expType not in ['Exp3','Exp4','Exp6']):
        return None
    
    distanceToMean = (trial['Continuous target']-trial['Mean target'])*360/nWheelSlices
    endPoint = 0.5*((360/nTargets)-(2*unusedAngle))*(1-(1/splitSections))
    startPoint = -endPoint
    allSections = numpy.linspace(startPoint,endPoint,splitSections)
    return allSections[numpy.argmin(numpy.abs(allSections-distanceToMean))]