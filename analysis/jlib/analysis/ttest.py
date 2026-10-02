def buildPairs(pairLabels):
    '''
    Internal function to convert user input into a list of pairs to be tested.
    '''
    pairs = ','.join('''list(i1="{}",i2="{}")'''.format(item[0],item[1]) for item in pairLabels)
    return pairs

def ttestIS(rPath='',data=None,vars=None,group=None,
            isStudent=None,isWelchs=None,isMann=None,hypothesis=None,
            effectSize=None,descriptives=None):
    '''
    Function to perform independent samples t-test on a dataset using R.
    '''
    import os, pandas
    os.environ['R_HOME'] = rPath
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri

    vars = ','.join(['''"{}"'''.format(item) for item in vars])
    group = ',group="{}"'.format(group) if group is not None else ""
    students = ',students={}'.format(str(isStudent).upper()) if isStudent is not None else ""
    welchs = ',welchs={}'.format(str(isWelchs).upper()) if isWelchs is not None else ""
    mann = ',mann={}'.format(str(isMann).upper()) if isMann is not None else ""
    hypothesis = ''',hypothesis="{}"'''.format(hypothesis) if hypothesis is not None else ""
    effectSize = ',effectSize={}'.format(str(effectSize).upper()) if effectSize is not None else ""
    desc = ',desc={}'.format(str(descriptives).upper()) if descriptives is not None else ""

    ttestParams = '''
        vars=list({})
        {}
        {}
        {}
        {}
        {}
        {}
        {}
    '''.format(vars,group,students,welchs,mann,hypothesis,effectSize,desc)

    with (robjects.default_converter + pandas2ri.converter).context():
        rData = robjects.conversion.get_conversion().py2rpy(data)
        robjects.globalenv['rData'] = rData

    fullTtestString = '''
        library(jmv)
        res.jmv.ttestIS <- ttestIS(data=rData,{})
        '''.format(ttestParams)
    fullTtest = robjects.r(fullTtestString)

    ttest = None
    normality = None
    descriptives = None

    with (robjects.default_converter + pandas2ri.converter).context():
        ttest = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestIS$ttest$asDF'''))
        normality = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestIS$assum$norm$asDF'''))
        descriptives = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestIS$desc$asDF'''))

    return {'ttest':ttest,'normality':normality,'descriptives':descriptives}

def ttestISFromArray(rPath='',group1=None,group2=None,
                     isStudent=None,isWelchs=None,isMann=None,hypothesis=None,
                     effectSize=None,descriptives=None):
    import pandas, numpy

    values = numpy.concatenate([group1,group2])
    groups = ['group1']*len(group1)+['group2']*len(group2)
    data = pandas.DataFrame({'value':values,'group':groups})

    return ttestIS(rPath,data,vars=['value'],group='group',
                   isStudent=isStudent,isWelchs=isWelchs,isMann=isMann,
                   hypothesis=hypothesis,effectSize=effectSize,descriptives=descriptives)

def ttestOS(rPath='',data=None,variables=None,
            testValue=None,isStudent=None,isWilcoxon=None,hypothesis=None,
            effectSize=None,descriptives=None):
    '''
    Function to perform one-sample t-test on a dataset using R.
    '''
    import os, pandas
    os.environ['R_HOME'] = rPath
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri

    variables = ','.join(['''"{}"'''.format(item) for item in variables])
    students = ',students={}'.format(str(isStudent).upper()) if isStudent is not None else ""
    wilcoxon = ',wilcoxon={}'.format(str(isWilcoxon).upper()) if isWilcoxon is not None else ""
    testValue = ',testValue={}'.format(testValue) if testValue is not None else ""
    hypothesis = ''',hypothesis="{}"'''.format(hypothesis) if hypothesis is not None else ""
    effectSize = ',effectSize={}'.format(str(effectSize).upper()) if effectSize is not None else ""
    desc = ',desc={}'.format(str(descriptives).upper()) if descriptives is not None else ""

    ttestParams = '''
        vars=list({})
        {}
        {}
        {}
        {}
        {}
        {}
    '''.format(variables,students,wilcoxon,testValue,hypothesis,effectSize,desc)

    with (robjects.default_converter + pandas2ri.converter).context():
        rData = robjects.conversion.get_conversion().py2rpy(data)
        robjects.globalenv['rData'] = rData

    fullTtestString = '''
        library(jmv)
        res.jmv.ttestOneS <- ttestOneS(data=rData,{})
        '''.format(ttestParams)
    # print(fullTtestString)
    fullTtest = robjects.r(fullTtestString)

    ttest = None
    normality = None
    descriptives = None

    with (robjects.default_converter + pandas2ri.converter).context():
        ttest = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestOneS$ttest$asDF'''))
        normality = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestOneS$normality$asDF'''))
        descriptives = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestOneS$descriptives$asDF'''))
    
    return {'ttest':ttest,'normality':normality,'descriptives':descriptives}

def ttestOSFromArray(rPath='',variables=None,
                     testValue=None,isStudent=None,isWilcoxon=None,hypothesis=None,
                     effectSize=None,descriptives=None):
    import pandas

    keys = ['set{}'.format(i) for i in list(range(len(variables)))]
    source = dict(zip(keys,variables))
    data = pandas.DataFrame(source)

    return ttestOS(rPath,data,keys,
            testValue,isStudent,isWilcoxon,hypothesis,
            effectSize,descriptives)

def ttestPS(rPath='',data=None,pairLabels=None,
            isStudent=None,isWilcoxon=None,hypothesis=None,
            effectSize=None,descriptives=None):
    '''
    Function to perform paired samples t-test on a dataset using R.
    '''
    import os, pandas
    os.environ['R_HOME'] = rPath    
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri

    pairs = buildPairs(pairLabels)
    students = ',students={}'.format(str(isStudent).upper()) if isStudent is not None else ""
    wilcoxon = ',wilcoxon={}'.format(str(isWilcoxon).upper()) if isWilcoxon is not None else ""
    hypothesis = ''',hypothesis="{}"'''.format(hypothesis) if hypothesis is not None else ""
    effectSize = ',effectSize={}'.format(str(effectSize).upper()) if effectSize is not None else ""
    desc = ',desc={}'.format(str(descriptives).upper()) if descriptives is not None else ""

    ttestParams = '''
        pairs=list({})
        {}
        {}
        {}
        {}
        {}
        '''.format(pairs,students,wilcoxon,hypothesis,effectSize,desc)

    with (robjects.default_converter + pandas2ri.converter).context():
        rData = robjects.conversion.get_conversion().py2rpy(data)
        robjects.globalenv['rData'] = rData

    fullTtestString = '''
        library(jmv)
        res.jmv.ttestPS <- ttestPS(data=rData,{})
        '''.format(ttestParams)
    #print(fullTtestString)
    fullTtest = robjects.r(fullTtestString)

    ttest = None
    normality = None
    descriptives = None

    with (robjects.default_converter + pandas2ri.converter).context():
        ttest = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestPS$ttest$asDF'''))
        normality = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestPS$norm$asDF'''))
        descriptives = robjects.conversion.get_conversion().rpy2py(robjects.r('''res.jmv.ttestPS$desc$asDF'''))
        
    return {'ttest':ttest,'normality':normality,'descriptives':descriptives}

def ttestPSFromArray(rPath='',column1=None,column2=None,
                     isStudent=None,isWilcoxon=None,hypothesis=None,
                     effectSize=None,descriptives=None):
    
    import pandas

    source = {'set1':column1,'set2':column2}
    data = pandas.DataFrame(source)
    
    return ttestPS(rPath,data,[['set1','set2']],
                   isStudent,isWilcoxon,hypothesis,
                   effectSize,descriptives)