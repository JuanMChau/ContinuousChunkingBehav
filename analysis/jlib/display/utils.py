from jlib.data import constants
import numpy

def displayPairwiseSignificance(ax,x,y,p,texty=None,size=10,color='black'):
    ax.plot(x,y,color=color)
    if (texty is None):
        ax.text(0.5*numpy.sum(x),y[0],constants.getAsteriskSignificance(p),
                size=size,color=color,horizontalalignment='center')
    else:
        ax.text(0.5*numpy.sum(x),texty,constants.getAsteriskSignificance(p),
                size=size,color=color,horizontalalignment='center')
    return None

def displayOneSampleSignificance(ax,x,y,p,size=10,color='black'):
    ax.text(x,y,constants.getAsteriskSignificance(p),
            size=size,color=color,horizontalalignment='center')
    return None