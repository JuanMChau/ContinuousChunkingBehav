def descriptiveHelper(badDataFrame,varNames):
    '''
    Helper function to parse descriptives table into an actual DF.
    '''
    import pandas
    
    sortedInfo = {}

    for name in varNames:
        sortedInfo[name] = []

    varCounter = 0
    currentVarName = None
    
    varSets = []
    currentVarSet = []

    for i in badDataFrame.columns:
        if (currentVarName is None):
            currentVarName = badDataFrame[i].to_numpy()[0]
            if (badDataFrame[i].to_numpy()[0] not in sortedInfo):
                sortedInfo[badDataFrame[i].to_numpy()[0]] = []
        else:
            if (varCounter==len(varNames)):
                sortedInfo[currentVarName].append(badDataFrame[i].to_numpy()[0])
                varCounter = 0
                currentVarName = None
                
                if (currentVarSet not in varSets):
                    varSets.append(currentVarSet)
                    for i in range(len(currentVarSet)):
                        sortedInfo[varNames[i]].append(currentVarSet[i])
                currentVarSet = []
            else:                
                currentVarSet.append(badDataFrame[i].to_numpy()[0])
                varCounter += 1

        print(sortedInfo)

    return pandas.DataFrame.from_dict(sortedInfo)

def getDescriptives(rPath='',data=None,formula=None,
                    N=False,missing=False,mean=True,median=False,mode=False,sum=False,
                    sd=True,shapiroWilk=True,debug=False):
    '''
    Function to calculate descriptives (including normality) using R.
    '''
    import os, pandas
    os.environ['R_HOME'] = rPath    
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri

    formula = 'formula={},'.format(formula) if formula is not None else ''
    N = 'T' if N else 'F'
    missing = 'T' if missing else 'F'
    mean = 'T' if mean else 'F'
    median = 'T' if median else 'F'
    mode = 'T' if mode else 'F'
    sum = 'T' if sum else 'F'
    sd = 'T' if sd else 'F'
    shapiroWilk = 'T' if shapiroWilk else 'F'

    descriptivesParams = '''
        n={},missing={},mean={},
        median={},mode={},sum={},
        sd={},sw={}
    '''.format(N,missing,mean,median,mode,sum,sd,shapiroWilk)

    with (robjects.default_converter + pandas2ri.converter).context():
        rData = robjects.conversion.get_conversion().py2rpy(data)
        robjects.globalenv['rData'] = rData

    fullDescriptivesString = '''
        library(jmv)
        res.jmv.desc <- descriptives({}rData,{})
    '''.format(formula,descriptivesParams)
    if (debug):
        print(fullDescriptivesString)
    fullDescriptives = robjects.r(fullDescriptivesString)

    descriptives = None

    with (robjects.default_converter + pandas2ri.converter).context():
        descriptives = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.desc$descriptives$asDF'''))

    return {'descriptives':descriptives}