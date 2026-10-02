import pandas, numpy

class MapValueHandler:
    def __init__(self,path):
        '''
        Class constructor.
        '''
        self.path = path
        self.data = self.loadData(path)
        self.keysModifiedThisSession = []

    def getValueFromKey(self,key):
        '''
        Method to return data stored within the class.
        '''
        if (key in self.data):
            return self.data[key]
        return None

    def loadData(self,path):
        '''
        Method to initialize the class at a specific location.
        '''
        try:
            rawData = pandas.read_csv(path)
            return dict(zip(rawData['Keys'],rawData['Values']))
        except:
            print(path+" not found, empty file will be created.")
            return dict()
    
    def saveData(self):
        '''
        Method to save the class contents at the specified location.
        '''
        dataToSave = pandas.DataFrame.from_dict(self.data,orient='index').reset_index()
        dataToSave = dataToSave.set_axis(['Keys','Values'],axis=1)
        try:
            dataToSave.to_csv(self.path,index=False)
        except:
            print(self.path+" not found, check for folder structure")
        print("All data saved correctly to "+self.path)

    def setKeyValue(self,key,value):
        '''
        Method to add or replace data stored within the class.
        '''
        if (key in self.keysModifiedThisSession):
            print("Attempting to save already stored key: "+key)
        else:
            self.data[key] = value
            self.keysModifiedThisSession.append(key)
