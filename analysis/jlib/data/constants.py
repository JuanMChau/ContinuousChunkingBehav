import numpy

P_VALUE_STRING = '{}{:.4f}'
P_VALUE_SMALL_THRESHOLD = 0.0001
ANOVA_STRING = '$F({:.0f},{:.0f})={:.4f}, p{}, {}$'
ANOVA_CORRECTED_STRING = '$F({:.2f},{:.2f})={:.4f}, p{}, {}$'
EFFECT_SIZE_STRING = '{}={:.2f}'
EFFECT_SIZE_STRING_SMALL = '{}<{:.2f}'
EFFECT_SIZE_SMALL_THRESHOLD = 0.01
EFFECT_SIZE_SYMBOLS = {'ges':'\eta^2_G','eta':'\eta^2','partEta':'\eta^2_p','omega':'\omega^2'}
FRIEDMAN_STRING = "$\chi^2({:.0f})={:.2f}, p{}, W={:.2f}$"
SPEARMAN_STRING = r"$\rho={:.3f}, p{}$"
TTEST_STRING = r"$t({:.0f})={:.4f}, p{}, \textit{{Cohen's d}}={:.2f}$"
WELCH_TTEST_STRING = r"$t({:.2f})={:.4f}, p{}, \textit{{Cohen's d}}={:.2f}$"
WILCOXON_STRING = "$W={:.0f}, p{}, r_{{rb}}={:.2f}$"
TIMEWINDOW_STRING = "${:.0f}-{:.0f} ms, p{}$"

def _createEffectSize(type,value):
    '''
    Function to create a string to report test effect sizes.
    '''
    if (value<EFFECT_SIZE_SMALL_THRESHOLD):
        return EFFECT_SIZE_STRING_SMALL.format(EFFECT_SIZE_SYMBOLS[type],EFFECT_SIZE_SMALL_THRESHOLD)
    return EFFECT_SIZE_STRING.format(EFFECT_SIZE_SYMBOLS[type],value)
    
def _createPValue(value):
    '''
    Function to create a string to report test p-values.
    '''
    if (value<P_VALUE_SMALL_THRESHOLD):
        return P_VALUE_STRING.format('<',P_VALUE_SMALL_THRESHOLD)
    return P_VALUE_STRING.format('=',value)

def _createAnovaResults(dof1,dof2,F,p,effectSizeType,effectSize):
    '''
    Function to create a string to report anova results.
    '''
    return ANOVA_STRING.format(dof1,dof2,F,_createPValue(p),_createEffectSize(effectSizeType,effectSize))

def _createAnovaCorrectedResults(dof1,dof2,F,p,effectSizeType,effectSize):
    '''
    Function to create a string to report anova results.
    '''
    return ANOVA_CORRECTED_STRING.format(dof1,dof2,F,_createPValue(p),_createEffectSize(effectSizeType,effectSize))

def createDescriptives(mean,std):
    '''
    Function to create a string to report variable descriptives.
    '''
    return None

def _createFriedmanResults(dof,chi,p,effectSize):
    '''
    Function to create a string to report non-parametric anova results.
    '''
    return FRIEDMAN_STRING.format(dof,chi,_createPValue(p),effectSize)

def _createSpearmanResults(rho,p):
    '''
    Function to create a string to report non-parametric anova results.
    '''
    return SPEARMAN_STRING.format(rho,_createPValue(p))

def _createTtestResults(dof,t,p,effectSize):
    '''
    Function to create a string to report ttest (parametric) results.
    '''
    return TTEST_STRING.format(dof,t,_createPValue(p),effectSize)

def _createWelchResults(dof,t,p,effectSize):
    '''
    Function to create a string to report ttest (welch) results.
    '''
    return WELCH_TTEST_STRING.format(dof,t,_createPValue(p),effectSize)

def _createWilcoxonResults(z,p,effectSize):
    '''
    Function to create a string to report ttest (wilcoxon) results.
    '''
    return WILCOXON_STRING.format(z,_createPValue(p),effectSize)

def _createTimeWindowResults(start,end,p):
    '''
    Function to create a string to report time window cluster results.
    '''
    return TIMEWINDOW_STRING.format(start,end,_createPValue(p))

def getAsteriskSignificance(pvalue):
    if (pvalue<=0.0001):
        return '****'
    elif (pvalue<=0.001):
        return '***'
    elif (pvalue<=0.01):
        return '**'
    elif (pvalue<=0.05):
        return '*'
    return 'ns'

def extractAnovaResults(table,row1,row2,effectSize,correction=''):
    if correction=='':
        return _createAnovaResults(table['df'].iloc[row1],table['df'].iloc[row2],
                                   table['F'].iloc[row1],table['p'].iloc[row1],
                                   effectSize,table[effectSize].iloc[row1])
    else:
        return _createAnovaCorrectedResults(table['df'+correction].iloc[row1],table['df'+correction].iloc[row2],
                                            table['F'+correction].iloc[row1],table['p'+correction].iloc[row1],
                                            effectSize,table[effectSize].iloc[row1])

def extractFriedmanResults(table,sampleSize,nMeasurements):
    return _createFriedmanResults(table['df'].iloc[0],table['stat'].iloc[0],
                                  table['p'].iloc[0],table['stat'].iloc[0]/(sampleSize*(nMeasurements-1)))

def extractTtestResults(table,row,correctionFactor=1):
    return _createTtestResults(table['df'].iloc[row],table['sta'].iloc[row],
                               numpy.min([1,table['p'].iloc[row]*correctionFactor]),
                               table['e'].iloc[row])

def extractWelchResults(table,row,correctionFactor=1):
    return _createWelchResults(table['df'].iloc[row],table['stat'].iloc[row],
                               numpy.min([1,table['p'].iloc[row]*correctionFactor]),
                               table['es'].iloc[row])

def extractWilcoxonResults(table,row,correctionFactor=1):
    return _createWilcoxonResults(table['stat'].iloc[row],
                                  numpy.min([1,table['p'].iloc[row]*correctionFactor]),
                                  table['es'].iloc[row])