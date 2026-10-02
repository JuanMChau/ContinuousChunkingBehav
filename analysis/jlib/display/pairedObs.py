import matplotlib.pyplot as pyplot
from seaborn import boxplot
import numpy

def boxPlotPaired(**kwargs):
    '''
    Function to display boxplot with paired observations at the same time.
    '''
    boxplot(hue=kwargs['x'],showfliers=False,**kwargs)

    data = kwargs['data']
    ax = kwargs['ax']

    linecolor = '#7a7c80'

    varCounter = 0
    jitter = []
    allPoints = []
    
    for condition in data[kwargs['x']].unique():
        dataPoints = data[data[kwargs['x']]==condition]
        allPoints.append(dataPoints.reset_index())
        jitter.append(numpy.random.uniform(low=-0.5*kwargs['width'],high=0.5*kwargs['width'],size=(dataPoints.shape[0],)))
        if (isinstance(kwargs['palette'],dict)):
            ax.plot(varCounter+jitter[varCounter],dataPoints[kwargs['y']],marker='o',
                    markerfacecolor=kwargs['palette'][list(kwargs['palette'].keys())[varCounter]],markeredgecolor=linecolor,
                    linestyle='None',zorder=1000)
        else:
            ax.plot(varCounter+jitter[varCounter],dataPoints[kwargs['y']],marker='o',
                    markerfacecolor=kwargs['palette'][varCounter],markeredgecolor=linecolor,
                    linestyle='None',zorder=1000)
        
        if(varCounter>=1):
            plotCounter = 0
            for subject in dataPoints['Subject'].unique():
                ax.plot([varCounter-1+jitter[varCounter-1][plotCounter],varCounter+jitter[varCounter][plotCounter]],
                        [allPoints[varCounter-1][kwargs['y']][plotCounter],allPoints[varCounter][kwargs['y']][plotCounter]],
                        marker='None',color=linecolor,linestyle = '-',zorder=999)
                plotCounter += 1

        varCounter += 1

def boxPlotPairedGrouped(grouping,**kwargs):
    '''
    Function to display boxplot with paired observations at the same time.
        This one takes an additional grouping parameter that generates pairings/groupings between
        specific mesurements.
    '''
    # Get relevant values
    ax = kwargs.get('ax',None)
    data = kwargs.get('data',None)
    width = kwargs.get('width',0.8)    

    # Create grouping column
    data[grouping[0]] = data[grouping[1]].astype(str).agg('\n'.join,axis=1)
    # Pass values to boxplot
    boxplot(x=grouping[0],hue=grouping[0],showfliers=False,**kwargs)

    order = kwargs.get('order',data[grouping[0]].unique())

    # Add jittered individual points and paired lines
    varCounter = 0
    jitter = []
    allPoints = []
    linecolor = '#7a7c80'
    pairingLimit = len(numpy.unique(data[grouping[1][-1]]))

    for condition in order:
        dataPoints = data[data[grouping[0]]==condition]
        allPoints.append(dataPoints.reset_index())
        jitter.append(numpy.random.uniform(low=-0.3*width,high=0.3*width,size=(dataPoints.shape[0],)))
        ax.plot(varCounter+jitter[varCounter],dataPoints[kwargs['y']],marker='o',
                markerfacecolor=kwargs['palette'][condition],markeredgecolor=linecolor,linestyle='None',zorder=1000)
        
        if((varCounter%pairingLimit)!=0):
            plotCounter = 0
            for subject in dataPoints['Subject'].unique():
                ax.plot([varCounter-1+jitter[varCounter-1][plotCounter],varCounter+jitter[varCounter][plotCounter]],
                        [allPoints[varCounter-1][kwargs['y']][plotCounter],allPoints[varCounter][kwargs['y']][plotCounter]],
                        marker='None',color=linecolor,linestyle = '-',zorder=999)
                plotCounter += 1
    
        varCounter += 1