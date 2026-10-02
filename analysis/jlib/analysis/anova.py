def buildPostHocs(postHoc):
    '''
    Internal function to convert user input into post-hoc tests.
    '''
    nPostHocs = len(postHoc) if postHoc is not None else 0
    postHocs = ''',list({})'''.format(','.join(item if type(item) is not list else 'c({})'.format(','.join(subitem for subitem in item)) for item in postHoc)) if postHoc is not None else ""
    return postHocs,nPostHocs

def buildRMCells(rmLabels=None,rmLists=None,rmMeasureName=''):
    '''
    Internal function to convert user input into repeated measures cells.
    '''
    modifiedRmLists = [[''.join('{}{}'.format(rmLabels[j],str(i))) for i in range(rmLists[j][0],
                                                                                     rmLists[j][1]+1)] if type(rmLists[j][0])==int else rmLists[j] for j in range(0,len(rmLists))]
    import itertools
    combinedCells = list(itertools.product(*modifiedRmLists))
    rmCells = ','.join('''list(measure="{}{}",cell=c("{}"))'''.format(''.join(item),rmMeasureName,'''","'''.join(item)) for item in combinedCells)
    return rmCells

def buildRMFactor(rmLabel='',rmLists=None):
    '''
    Internal function to convert user input into a repeated measures factor.
    '''
    if (type(rmLists[0])==int):
        rmLevels = ''','''.join('''"{}{}"'''.format(rmLabel,item) for item in range(rmLists[0],
                                                                                    rmLists[1]+1))
    else:
        rmLevels = ','.join('''"{}"'''.format(item) for item in rmLists)        

    rmFactor = '''
        list(label='{}',
        levels=c({}))
        '''.format(rmLabel,rmLevels)
    
    return rmFactor

def anovaRM(rPath='',data=None,
            rmLabels=None,rmLists=None,rmMeasureName='',rmTerms='',
            bs=None,bsTerms=None,leveneTest=False,sphericityTest=False,
            effectSize=None,emMeans=None,postHocReq=None,postHocCorr='none',debug=False):
    '''
    Function to perform parametric, repeated measures ANOVA on a dataset using R.
    '''
    import os, pandas
    os.environ['R_HOME'] = rPath    
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri

    rmParams = ','.join(buildRMFactor(rmLabels[i],rmLists[i]) for i in range(0,len(rmLabels)))
    rmCells = buildRMCells(rmLabels,rmLists,rmMeasureName)
    bs = ''',bs=c("{}")'''.format('''","'''.join(bs)) if bs is not None else ""
    bsTerms = ''',bsTerms=list("{}")'''.format('''","'''.join(bsTerms)) if bsTerms is not None else ""
    effectSize = ''',effectSize=c("{}")'''.format('''","'''.join(effectSize)) if effectSize is not None else ""
    postHocCorr = ''',postHocCorr={}'''.format(postHocCorr) if postHocReq is not None else ""
    postHoc,nPostHoc = buildPostHocs(postHocReq)
    leveneTest = ',leveneTest=T' if leveneTest else ''
    sphericityTest = ',spherTests=T' if sphericityTest else ''
    sphericityCorrection = ',spherCorr=c("none", "GG", "HF")' if sphericityTest else ''
    nEmMeans = len(emMeans) if emMeans is not None else 0
    emMeans = ',emmTables=T,emMeans=c("{}")'.format('''","'''.join(emMeans)) if emMeans is not None else ''

    anovaParams = '''
        rm=list({}),
        rmCells=list({}),
        rmTerms={}
        {}
        {}
        {}
        {}
        {}
        {}{}{}{}
        '''.format(rmParams,rmCells,rmTerms,bs,bsTerms,effectSize,emMeans,
                   postHoc,postHocCorr,leveneTest,sphericityTest,sphericityCorrection)
    
    with (robjects.default_converter + pandas2ri.converter).context():
        rData = robjects.conversion.get_conversion().py2rpy(data)
        robjects.globalenv['rData'] = rData
        
    fullAnovaString = '''
        library(jmv)
        res.jmv.aov <- anovaRM(data=rData,{})
        '''.format(anovaParams)
    if (debug):
        print(fullAnovaString)
    fullAnova = robjects.r(fullAnovaString)

    withinSubjects = None
    betweenSubjects = None
    postHocs = None if nPostHoc==0 else []
    levene = None
    sphericity = None
    emMeanTables = None if emMeans=='' else []
    
    with (robjects.default_converter + pandas2ri.converter).context():
        withinSubjects = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aov$rmTable$asDF'''))
        betweenSubjects = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aov$bsTable$asDF'''))
        if (leveneTest!=''):
            levene = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aov$assump$leveneTable$asDF'''))
        if (sphericityTest!=''):
            sphericity = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aov$assump$spherTable$asDF'''))
        if (emMeans!=''):
            for i in list(range(nEmMeans)):
                emMeanTables.append(robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aov$emm[[{}]]$emmTable$asDF'''.format(i+1))))
        if (nPostHoc!=0):
            for i in list(range(nPostHoc)):
                postHocs.append(robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aov$postHoc[[{}]]$asDF'''.format(i+1))))

    output = {'within':withinSubjects,'between':betweenSubjects}
    if (leveneTest!=''):
        output['levene'] = levene
    if (sphericityTest!=''):
        output['sphericity'] = sphericity
    if (nPostHoc!=0):
        output['postHoc'] = postHocs
    if (emMeans!=''):
        output['emMeans'] = emMeanTables

    return output

def anovaRMNP(rPath='',data=None,measurements=None,pairs=False,debug=False):
    '''
    Function to perform non-parametric repeated measures ANOVA on a dataset using R.
    '''
    import os, pandas
    os.environ['R_HOME'] = rPath    
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri

    measurements = ','.join(measurements)
    pairs = ',pairs=TRUE' if pairs else ''

    anovaParams = '''
        measures=vars({})
        {}
        '''.format(measurements,pairs)
    
    with (robjects.default_converter + pandas2ri.converter).context():
        rData = robjects.conversion.get_conversion().py2rpy(data)
        robjects.globalenv['rData'] = rData
    
    fullAnovaString = '''
        library(jmv)
        res.jmv.aovNP <- anovaRMNP(data=rData,{})
        '''.format(anovaParams)
    if (debug):
        print(fullAnovaString)
    fullAnova = robjects.r(fullAnovaString)

    withinSubjects = None

    with (robjects.default_converter + pandas2ri.converter).context():
        withinSubjects = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aovNP$table$asDF'''))
        if (pairs!=''):
            postHocs = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.aovNP$comp$asDF'''))

    output = {'within':withinSubjects}
    if (pairs!=''):
        output['postHoc'] = postHocs

    return output