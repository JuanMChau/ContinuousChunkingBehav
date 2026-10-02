import re
import mne
import numpy

def preprocessETData(rawDataET,pathDataET,messageTemplate="",valueMap=None):
    startTimeStamp = None
    # Create an empty stimulus channel
    stimChanInfo = mne.create_info(['STI'],rawDataET.info["sfreq"],["stim"])
    stimChanRaw = mne.io.RawArray(numpy.zeros((1,len(rawDataET.times))),stimChanInfo)
    rawDataET.add_channels([stimChanRaw],force_update_info=True)
    # Create an empty event array
    eventArray = []
    # Populate event array with parsed information
    with open(pathDataET) as fileET:
        for line in fileET:
            # Find starting time value offset
            if (startTimeStamp is None):
                potentialTimeStamp = re.search("^[0-9]+",line)
                if (potentialTimeStamp):
                    startTimeStamp = int(potentialTimeStamp.group(0))
                    print(f"Found start time: {startTimeStamp}")
            # Parse all events that match the message template
            if (startTimeStamp is not None):
                potentialMessage = re.search(messageTemplate,line)
                if (potentialMessage):
                    timeStamp = re.search("[0-9]+",line)
                    messageTimeStamp = int(timeStamp.group(0))

                    splitLine = line.split(" ")
                    messageType = splitLine[-2]
                    messageValue = valueMap[messageType]+int(splitLine[-1])
                    
                    rawDataET.add_events([[messageTimeStamp-startTimeStamp,0,messageValue]])

    return rawDataET