import numpy
import mne

def removeBadEvents(mneRaw):
    '''
    Function to remove random triggers appearing on data
    '''

    # Get the trigger channel index
    stimChannelIndex = mne.channel_indices_by_type(mneRaw.info,picks=['stim'])    
    # Get the trigger channel
    stimChannel = mneRaw.copy()
    stimChannel.pick(['stim'])
    # Get the event list
    eventList = mne.event._find_events(stimChannel.get_data(),mneRaw.first_samp)
    # Find the points at which the event list is inconsistent
    eventIndices = numpy.argwhere(numpy.diff(eventList[:,0]) < 2)
    # Zero them out
    stimData = stimChannel.get_data()
    print(stimChannel._data)
    for index in eventIndices:
        stimData[0,eventList[index[0],0]] = 0
    # Replace the channel into the original structure
    mneRaw._data[stimChannelIndex['stim'],:] = stimData
    return mneRaw