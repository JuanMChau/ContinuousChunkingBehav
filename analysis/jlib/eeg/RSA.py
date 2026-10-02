import numpy, multiprocessing
from joblib import Parallel, delayed

def _halfSplitMahaDistanceRSA(eegData,behavData,weakestSampleSize,startSample,endSample,variableToSplit,
                              splitType=None,subsampleSeed=1):
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import LedoitWolf

    # Initialize decoding outputs
    globalDecoding = []
    globalMahaDistances = []

    for subsample in range(startSample,endSample):
        print(f"Processing subsample {subsample}")
        # Set rng seed for control/replication
        rng = numpy.random.default_rng(subsampleSeed+subsample)        
        # Get subsample indices
        subsampleIndices = behavData.groupby(variableToSplit).sample(weakestSampleSize,random_state=rng).index
        # Create the subsample
        behavDataSubsample = behavData.loc[subsampleIndices,:].copy().reset_index()
        eegDataSubsample = eegData[subsampleIndices].copy()
        
        # Initialize behavioural indices
        behavIndexEven = []
        behavIndexOdd = []
        # Populate behavioural indices
        if (splitType is None):
            # Split the data equally based on all conditions
            for condition in numpy.unique(behavDataSubsample[variableToSplit]):
                # Get the trials that belong to each individual condition
                indexItem = behavDataSubsample.index[behavDataSubsample[variableToSplit]==condition].tolist()
                # Do a half split on them
                behavIndexEven += indexItem[0::2]
                behavIndexOdd += indexItem[1::2]
        else:
            # Split the data equally based on grouped conditions
            for conditionGroup in splitType:
                for condition in conditionGroup:
                    # Get the trials that belong to each individual condition
                    indexItem = behavDataSubsample.index[behavDataSubsample[variableToSplit]==condition].tolist()
                    # Do a half split on them
                    behavIndexEven += indexItem[0::2]
                    behavIndexOdd += indexItem[1::2]
        # Sort indices to avoid weird behaviour
        behavIndexEven.sort()
        behavIndexOdd.sort()

        # Split the behavioural data in two parts
        behavEven = behavDataSubsample.iloc[behavIndexEven].copy().reset_index(drop=True)
        behavOdd = behavDataSubsample.iloc[behavIndexOdd].copy().reset_index(drop=True)
        # Split the EEG data in two parts
        eegEven = eegDataSubsample[behavIndexEven].copy()
        eegOdd = eegDataSubsample[behavIndexOdd].copy()

        # Create residual arrays
        residualEven = eegEven.copy()
        residualOdd = eegOdd.copy()
        # Calculate residuals
        if (splitType is None):
            # Calculate split-half averages for each condition to be considered
            for condition in numpy.unique(behavEven[variableToSplit]):
                # Split-half averages for odd trials
                indicesEven = behavEven[behavEven[variableToSplit]==condition].index.array.tolist()
                exec(f"erp_{condition}_0 = eegEven[indicesEven].average().data")
                # Split-half averages for even trials
                indicesOdd = behavOdd[behavOdd[variableToSplit]==condition].index.array.tolist()
                exec(f"erp_{condition}_1 = eegOdd[indicesOdd].average().data")

            # Calculate residuals for even trials
            for index, epoch in enumerate(residualEven):
                exec(f"residualEven._data[index] = epoch-erp_{behavEven.iloc[index][variableToSplit]}_0")
            # Calculate residuals for odd trials
            for index, epoch in enumerate(residualOdd):
                exec(f"residualOdd._data[index] = epoch-erp_{behavOdd.iloc[index][variableToSplit]}_1")
        else:
            # Calculate split-half averages for each condition group to be considered
            for conditionGroupIndex in list(range(len(splitType))):
                # Split-half averages for odd trials
                indicesEven = behavEven[behavEven[variableToSplit].isin(splitType[conditionGroupIndex])].index.array.tolist()
                exec(f"erp_{conditionGroupIndex}_0 = eegEven[indicesEven].average().data")
                # Split-half averages for even trials
                indicesOdd = behavOdd[behavOdd[variableToSplit].isin(splitType[conditionGroupIndex])].index.array.tolist()
                exec(f"erp_{conditionGroupIndex}_1 = eegOdd[indicesOdd].average().data")
        
            # Calculate residuals for even trials
            for index, epoch in enumerate(residualEven):
                conditionGroupIndex = numpy.argwhere(numpy.array(splitType)==behavEven.iloc[index][variableToSplit])[0][0]
                exec(f"residualEven._data[index] = epoch-erp_{conditionGroupIndex}_0")
            # Calculate residuals for odd trials
            for index, epoch in enumerate(residualOdd):
                conditionGroupIndex = numpy.argwhere(numpy.array(splitType)==behavOdd.iloc[index][variableToSplit])[0][0]
                exec(f"residualOdd._data[index] = epoch-erp_{conditionGroupIndex}_1")
        
        # Extract residual data
        residualEvenRaw = residualEven.get_data(copy=False)
        residualOddRaw = residualOdd.get_data(copy=False)
        # Initialize covariances
        covariances = []
        # Initiate LW estimator
        LWEven = LedoitWolf()
        LWOdd = LedoitWolf()
        # Calculate covariances for each time point
        for t in range(residualEvenRaw.shape[2]):
            # Estimate the covariances at that point
            tCovarianceEven = LWEven.fit(residualEvenRaw[:,:,t]).covariance_
            tCovarianceOdd = LWOdd.fit(residualOddRaw[:,:,t]).covariance_
            # Save it at the big matrix
            covariances.append(0.5*(tCovarianceEven+tCovarianceOdd))
        
        # Get a list of conditions
        if (splitType is None):
            conditions = numpy.unique(behavEven[variableToSplit])
        else:
            conditions = list(range(len(splitType)))
        # Initialize mahalanobis distances for each pair of conditions
        mahaDistances = {f"{condition1}_{index}-{condition2}_{1-index}": numpy.zeros(eegEven.times.size)
                        for condition1 in conditions for condition2 in conditions for index in [0,1]}
        # Calculate all mahalanobis distances
        for t in range(len(covariances)):
            # Invert covariance matrix
            tInverseCovariance = numpy.linalg.inv(covariances[t])
            for condition1 in conditions:
                for condition2 in conditions:
                    # Calculate the difference between condition means - even vs odd
                    meanDifference = eval(f"erp_{condition1}_0[:,t]-erp_{condition2}_1[:,t]")
                    # Calculate the mahalanobis distance
                    mahaDistance = mahalanobis(meanDifference,numpy.zeros_like(meanDifference),tInverseCovariance)
                    # Store the mahalanobis distance
                    mahaDistances[f"{condition1}_0-{condition2}_1"][t] = mahaDistance

                    # Calculate the difference between condition means - odd vs even
                    meanDifference = eval(f"erp_{condition1}_1[:,t]-erp_{condition2}_0[:,t]")
                    # Calculate the mahalanobis distance
                    mahaDistance = mahalanobis(meanDifference,numpy.zeros_like(meanDifference),tInverseCovariance)
                    # Store the mahalanobis distance
                    mahaDistances[f"{condition1}_1-{condition2}_0"][t] = mahaDistance
        
        # Initialize decoding
        decoding = numpy.zeros(mahaDistances[f"{conditions[0]}_0-{conditions[0]}_1"].shape)
        # Calculate decoding from mahalanobis distances
        for condition1 in conditions:
            # Subtract the distances between same-condition trials
            decoding -= mahaDistances[f"{condition1}_0-{condition1}_1"]/2
            decoding -= mahaDistances[f"{condition1}_1-{condition1}_0"]/2
            # Add the distances between different-condition trials
            for condition2 in conditions:
                if condition1 != condition2:
                    decoding += mahaDistances[f"{condition1}_0-{condition2}_1"]/(2*(len(conditions)-1))
                    decoding += mahaDistances[f"{condition1}_1-{condition2}_0"]/(2*(len(conditions)-1))
        
        globalMahaDistances.append(mahaDistances)
        globalDecoding.append(decoding)
    
    return globalMahaDistances,globalDecoding

def halfSplitMahaDistanceRSA(eegData,behavData,variableToSplit,splitType=None,subsamples=1,batchSize=1,nJobs=-1,
                             subsamplesSeed=1):
    # Set rng seed for control/replication
    rng = numpy.random.default_rng(subsamplesSeed)

    # Calculate the smallest sample size per condition
    variableCounts = behavData.value_counts(variableToSplit)
    weakestSampleSize = numpy.min(variableCounts)

    # Calculate the number of batches to be executed
    batches = subsamples//batchSize
    print("Batches: ",batches," size: ",batchSize)
    
    results = Parallel(n_jobs=nJobs,verbose=10)(delayed(_halfSplitMahaDistanceRSA)(eegData,behavData,weakestSampleSize,
                                                                                   i*batchSize,(i+1)*batchSize,
                                                                                   variableToSplit,splitType,
                                                                                   subsamplesSeed) for i in range(batches))

    globalMahaDistances, globalDecoding = zip(*results)
    globalDecoding = numpy.concatenate([i for i in globalDecoding])
    globalMahaDistances = [i for j in globalMahaDistances for i in j]

    return globalMahaDistances,globalDecoding

def _slidingWindowHalfSplitMahaDistanceRSA(eegData,behavData,weakestSampleSize,startSample,endSample,
                                           variableToSplit,splitType=None,slidingWindowSize=1,subsampleSeed=1):
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import LedoitWolf

    # Initialize decoding outputs
    globalDecoding = []
    globalMahaDistances = []

    for subsample in range(startSample,endSample):
        print(f"Processing subsample {subsample}")
        # Set rng seed for control/replication
        rng = numpy.random.default_rng(subsampleSeed+subsample)        
        # Get subsample indices
        subsampleIndices = behavData.groupby(variableToSplit).sample(weakestSampleSize,random_state=rng).index
        # Create the subsample
        behavDataSubsample = behavData.loc[subsampleIndices,:].copy().reset_index()
        eegDataSubsample = eegData[subsampleIndices].copy()
        
        # Initialize behavioural indices
        behavIndexEven = []
        behavIndexOdd = []
        # Populate behavioural indices
        if (splitType is None):
            # Split the data equally based on all conditions
            for condition in numpy.unique(behavDataSubsample[variableToSplit]):
                # Get the trials that belong to each individual condition
                indexItem = behavDataSubsample.index[behavDataSubsample[variableToSplit]==condition].tolist()
                # Do a half split on them
                behavIndexEven += indexItem[0::2]
                behavIndexOdd += indexItem[1::2]
        else:
            # Split the data equally based on grouped conditions
            for conditionGroup in splitType:
                for condition in conditionGroup:
                    # Get the trials that belong to each individual condition
                    indexItem = behavDataSubsample.index[behavDataSubsample[variableToSplit]==condition].tolist()
                    # Do a half split on them
                    behavIndexEven += indexItem[0::2]
                    behavIndexOdd += indexItem[1::2]
        # Sort indices to avoid weird behaviour
        behavIndexEven.sort()
        behavIndexOdd.sort()

        # Split the behavioural data in two parts
        behavEven = behavDataSubsample.iloc[behavIndexEven].copy().reset_index(drop=True)
        behavOdd = behavDataSubsample.iloc[behavIndexOdd].copy().reset_index(drop=True)
        # Split the EEG data in two parts
        eegEven = eegDataSubsample[behavIndexEven].copy()
        eegOdd = eegDataSubsample[behavIndexOdd].copy()

        # Create residual arrays
        residualEven = eegEven.copy()
        residualOdd = eegOdd.copy()
        # Calculate residuals
        if (splitType is None):
            # Calculate split-half averages for each condition to be considered
            for condition in numpy.unique(behavEven[variableToSplit]):
                # Split-half averages for odd trials
                indicesEven = behavEven[behavEven[variableToSplit]==condition].index.array.tolist()
                exec(f"erp_{condition}_0 = eegEven[indicesEven].average().data")
                # Split-half averages for even trials
                indicesOdd = behavOdd[behavOdd[variableToSplit]==condition].index.array.tolist()
                exec(f"erp_{condition}_1 = eegOdd[indicesOdd].average().data")

            # Calculate residuals for even trials
            for index, epoch in enumerate(residualEven):
                exec(f"residualEven._data[index] = epoch-erp_{behavEven.iloc[index][variableToSplit]}_0")
            # Calculate residuals for odd trials
            for index, epoch in enumerate(residualOdd):
                exec(f"residualOdd._data[index] = epoch-erp_{behavOdd.iloc[index][variableToSplit]}_1")
        else:
            # Calculate split-half averages for each condition group to be considered
            for conditionGroupIndex in list(range(len(splitType))):
                # Split-half averages for odd trials
                indicesEven = behavEven[behavEven[variableToSplit].isin(splitType[conditionGroupIndex])].index.array.tolist()
                exec(f"erp_{conditionGroupIndex}_0 = eegEven[indicesEven].average().data")
                # Split-half averages for even trials
                indicesOdd = behavOdd[behavOdd[variableToSplit].isin(splitType[conditionGroupIndex])].index.array.tolist()
                exec(f"erp_{conditionGroupIndex}_1 = eegOdd[indicesOdd].average().data")
        
            # Calculate residuals for even trials
            for index, epoch in enumerate(residualEven):
                conditionGroupIndex = numpy.argwhere(numpy.array(splitType)==behavEven.iloc[index][variableToSplit])[0][0]
                exec(f"residualEven._data[index] = epoch-erp_{conditionGroupIndex}_0")
            # Calculate residuals for odd trials
            for index, epoch in enumerate(residualOdd):
                conditionGroupIndex = numpy.argwhere(numpy.array(splitType)==behavOdd.iloc[index][variableToSplit])[0][0]
                exec(f"residualOdd._data[index] = epoch-erp_{conditionGroupIndex}_1")
        
        # Extract residual data
        residualEvenRaw = residualEven.get_data(copy=False)
        residualOddRaw = residualOdd.get_data(copy=False)
        # Initialize covariances
        covariances = []
        # Initiate LW estimator
        LWEven = LedoitWolf()
        LWOdd = LedoitWolf()
        # Calculate covariances for each time point
        for t in range(slidingWindowSize,residualEvenRaw.shape[2]+1):
            # Estimate the covariances at that point
            tCovarianceEven = LWEven.fit(residualEvenRaw[:,:,t-slidingWindowSize:t].reshape(residualEvenRaw.shape[0],-1)).covariance_
            tCovarianceOdd = LWOdd.fit(residualOddRaw[:,:,t-slidingWindowSize:t].reshape(residualOddRaw.shape[0],-1)).covariance_
            # Save it at the big matrix
            covariances.append(0.5*(tCovarianceEven+tCovarianceOdd))
        
        # Get a list of conditions
        if (splitType is None):
            conditions = numpy.unique(behavEven[variableToSplit])
        else:
            conditions = list(range(len(splitType)))
        # Initialize mahalanobis distances for each pair of conditions
        mahaDistances = {f"{condition1}_{index}-{condition2}_{1-index}": numpy.zeros(eegEven.times.size)
                        for condition1 in conditions for condition2 in conditions for index in [0,1]}
        # Calculate all mahalanobis distances
        for t in range(len(covariances)):
            # Invert covariance matrix
            tInverseCovariance = numpy.linalg.inv(covariances[t])
            for condition1 in conditions:
                for condition2 in conditions:
                    # Calculate the difference between condition means - even vs odd
                    meanDifference = eval(f"erp_{condition1}_0[:,t:t+slidingWindowSize].flatten()-erp_{condition2}_1[:,t:t+slidingWindowSize].flatten()")
                    # Calculate the mahalanobis distance
                    mahaDistance = mahalanobis(meanDifference,numpy.zeros_like(meanDifference),tInverseCovariance)
                    # Store the mahalanobis distance
                    mahaDistances[f"{condition1}_0-{condition2}_1"][t+slidingWindowSize-1] = mahaDistance

                    # Calculate the difference between condition means - odd vs even
                    meanDifference = eval(f"erp_{condition1}_1[:,t:t+slidingWindowSize].flatten()-erp_{condition2}_0[:,t:t+slidingWindowSize].flatten()")
                    # Calculate the mahalanobis distance
                    mahaDistance = mahalanobis(meanDifference,numpy.zeros_like(meanDifference),tInverseCovariance)
                    # Store the mahalanobis distance
                    mahaDistances[f"{condition1}_1-{condition2}_0"][t+slidingWindowSize-1] = mahaDistance
        
        # Initialize decoding
        decoding = numpy.zeros(mahaDistances[f"{conditions[0]}_0-{conditions[0]}_1"].shape)
        # Calculate decoding from mahalanobis distances
        for condition1 in conditions:
            # Subtract the distances between same-condition trials
            decoding -= mahaDistances[f"{condition1}_0-{condition1}_1"]/2
            decoding -= mahaDistances[f"{condition1}_1-{condition1}_0"]/2
            # Add the distances between different-condition trials
            for condition2 in conditions:
                if condition1 != condition2:
                    decoding += mahaDistances[f"{condition1}_0-{condition2}_1"]/(2*(len(conditions)-1))
                    decoding += mahaDistances[f"{condition1}_1-{condition2}_0"]/(2*(len(conditions)-1))
        
        globalMahaDistances.append(mahaDistances)
        globalDecoding.append(decoding)
    
    return globalMahaDistances,globalDecoding

def slidingWindowHalfSplitMahaDistanceRSA(eegData,behavData,variableToSplit,splitType=None,slidingWindowSize=1,
                                          subsamples=1,batchSize=1,nJobs=-1,subsamplesSeed=1):
    # Set rng seed for control/replication
    rng = numpy.random.default_rng(subsamplesSeed)

    # Calculate the smallest sample size per condition
    variableCounts = behavData.value_counts(variableToSplit)
    weakestSampleSize = numpy.min(variableCounts)

    # Calculate the number of batches to be executed
    batches = subsamples//batchSize
    print("Batches: ",batches," size: ",batchSize)
    
    results = Parallel(n_jobs=nJobs,
                       verbose=10)(delayed(_slidingWindowHalfSplitMahaDistanceRSA)(eegData,behavData,weakestSampleSize,
                                                                                   i*batchSize,(i+1)*batchSize,
                                                                                   variableToSplit,
                                                                                   splitType,slidingWindowSize,
                                                                                   subsamplesSeed) for i in range(batches))

    globalMahaDistances, globalDecoding = zip(*results)
    globalDecoding = numpy.concatenate([i for i in globalDecoding])
    globalMahaDistances = [i for j in globalMahaDistances for i in j]

    return globalMahaDistances,globalDecoding

def _slidingWindowMahaDistanceRSA(eegData,behavData,weakestSamplesize,startSample,endSample,variableToSplit,
                                  splitType,slidingWindowSize=1,subsampleSeed=1):
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import LedoitWolf

    # Initialize decoding outputs
    globalDecoding = []
    globalMahaDistances = []

    # Create array of supported maha distances
    closeDistances = []
    farDistances = []

    # Create indices for differences to be calculated
    for item in splitType:
        for closeAssociate in item[1]:
            if ([item[0],closeAssociate] not in closeDistances):
                closeDistances.append([item[0],closeAssociate])
        for farAssociate in item[2]:
            if ([item[0],farAssociate] not in farDistances):
                farDistances.append([item[0],farAssociate])

    for subsample in range(startSample,endSample):
        print(f"Processing subsample {subsample}")
        # Set rng seed for control/replication
        rng = numpy.random.default_rng(subsampleSeed+subsample)        
        # Get subsample indices
        subsampleIndices = behavData.groupby(variableToSplit).sample(weakestSamplesize,random_state=rng).index
        # Create the subsample
        behavDataSubsample = behavData.loc[subsampleIndices,:].copy().reset_index()
        eegDataSubsample = eegData[subsampleIndices].copy()

        # Create residual arrays
        residuals = eegDataSubsample.copy()
        # Calculate averages for each condition to be considered
        for condition in numpy.unique(behavDataSubsample[variableToSplit]):
            # Calculate averages for odd trials
            indices = behavDataSubsample[behavDataSubsample[variableToSplit]==condition].index.array.tolist()
            exec(f"erp_{condition} = eegDataSubsample[indices].average().data")

        # Calculate residuals
        for index, epoch in enumerate(residuals):
            exec(f"residuals._data[index] = epoch-erp_{behavDataSubsample.iloc[index][variableToSplit]}")
        
        # Extract residual data
        residualsRaw = residuals.get_data(copy=False)
        # Initialize covariances
        covariances = []
        # Initiate LW estimator
        LW = LedoitWolf()
        # Calculate covariances for each time point
        for t in range(slidingWindowSize,residualsRaw.shape[2]+1):
            # Estimate the covariances at that point
            tCovariance = LW.fit(residualsRaw[:,:,t-slidingWindowSize:t].reshape(residualsRaw.shape[0],-1)).covariance_
            # Save it at the big matrix
            covariances.append(tCovariance)
        
        # Get a list of conditions
        conditions = numpy.unique(behavDataSubsample[variableToSplit])
        # Initialize mahalanobis distances for each pair of conditions
        mahaDistances = {f"{condition1}-{condition2}": numpy.zeros(eegDataSubsample.times.size)
                        for condition1 in conditions for condition2 in conditions}
        # Calculate all mahalanobis distances
        for t in range(len(covariances)):
            # Invert covariance matrix
            tInverseCovariance = numpy.linalg.inv(covariances[t])
            for condition1 in conditions:
                for condition2 in conditions:
                    # Calculate the difference between condition means
                    meanDifference = eval(f"erp_{condition1}[:,t:t+slidingWindowSize].flatten()-erp_{condition2}[:,t:t+slidingWindowSize].flatten()")
                    # Calculate the mahalanobis distance
                    mahaDistance = mahalanobis(meanDifference,numpy.zeros_like(meanDifference),tInverseCovariance)
                    # Store the mahalanobis distance
                    mahaDistances[f"{condition1}-{condition2}"][t+slidingWindowSize-1] = mahaDistance
        
        # Initialize decoding
        decoding = numpy.zeros(mahaDistances[f"{conditions[0]}-{conditions[0]}"].shape)
        # Calculate decoding from mahalanobis distances

        for condition1 in conditions.tolist():
            for condition2 in conditions.tolist():
                # Add the distances between conditions
                if [condition1,condition2] in closeDistances:
                    decoding -= mahaDistances[f"{condition1}-{condition2}"]/(2*len(closeDistances))
                # Subtract the distances between same-condition trials
                if [condition1,condition2] in farDistances:
                    decoding += mahaDistances[f"{condition1}-{condition2}"]/(2*len(farDistances))
                    
        globalMahaDistances.append(mahaDistances)
        globalDecoding.append(decoding)
    
    return globalMahaDistances,globalDecoding

def slidingWindowMahaDistanceRSA(eegData,behavData,variableToSplit,splitType,slidingWindowSize=1,subsamples=1,batchSize=1,
                                 nJobs=-1,subsamplesSeed=1):
    # Set rng seed for control/replication
    rng = numpy.random.default_rng(subsamplesSeed)

    # Calculate the smallest sample size per condition
    variableCounts = behavData.value_counts(variableToSplit)
    weakestSamplesize = numpy.min(variableCounts)

    # Calculate the number of batches to be executed
    batches = subsamples//batchSize
    print("Batches: ",batches," size: ",batchSize)

    results = Parallel(n_jobs=nJobs,verbose=10)(delayed(_slidingWindowMahaDistanceRSA)(eegData,behavData,
                                                                                       weakestSamplesize,
                                                                                       i*batchSize,(i+1)*batchSize,
                                                                                       variableToSplit,
                                                                                       splitType,slidingWindowSize,
                                                                                       subsamplesSeed) for i in range(batches))
    
    globalMahaDistances, globalDecoding = zip(*results)
    globalDecoding = numpy.concatenate([i for i in globalDecoding])
    globalMahaDistances = [i for j in globalMahaDistances for i in j]

    return globalMahaDistances,globalDecoding

def _mahaDistanceRSA(eegData,behavData,weakestSampleSize,startSample,endSample,variableToSplit,splitType,
                     subsampleSeed=1):
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import LedoitWolf

    # Initialize decoding outputs
    globalDecoding = []
    globalMahaDistances = []

    # Create array of supported maha distances
    closeDistances = []
    farDistances = []

    # Create indices for differences to be calculated
    for item in splitType:
        for closeAssociate in item[1]:
            if ([item[0],closeAssociate] not in closeDistances):
                closeDistances.append([item[0],closeAssociate])
        for farAssociate in item[2]:
            if ([item[0],farAssociate] not in farDistances):
                farDistances.append([item[0],farAssociate])

    for subsample in range(startSample,endSample):
        print(f"Processing subsample {subsample}")
        # Set rng seed for control/replication
        rng = numpy.random.default_rng(subsampleSeed+subsample)        
        # Get subsample indices
        subsampleIndices = behavData.groupby(variableToSplit).sample(weakestSampleSize,random_state=rng).index
        # Create the subsample
        behavDataSubsample = behavData.loc[subsampleIndices,:].copy().reset_index()
        eegDataSubsample = eegData[subsampleIndices].copy()

        # Create residual arrays
        residuals = eegDataSubsample.copy()
        # Calculate averages for each condition to be considered
        for condition in numpy.unique(behavDataSubsample[variableToSplit]):
            # Calculate averages for odd trials
            indices = behavDataSubsample[behavDataSubsample[variableToSplit]==condition].index.array.tolist()
            exec(f"erp_{condition} = eegDataSubsample[indices].average().data")

        # Calculate residuals
        for index, epoch in enumerate(residuals):
            exec(f"residuals._data[index] = epoch-erp_{behavDataSubsample.iloc[index][variableToSplit]}")
        
        # Extract residual data
        residualsRaw = residuals.get_data(copy=False)
        # Initialize covariances
        covariances = []
        # Initiate LW estimator
        LW = LedoitWolf()
        # Calculate covariances for each time point
        for t in range(residualsRaw.shape[2]):
            # Estimate the covariances at that point
            tCovariance = LW.fit(residualsRaw[:,:,t]).covariance_
            # Save it at the big matrix
            covariances.append(tCovariance)
        
        # Get a list of conditions
        conditions = numpy.unique(behavDataSubsample[variableToSplit])
        # Initialize mahalanobis distances for each pair of conditions
        mahaDistances = {f"{condition1}-{condition2}": numpy.zeros(eegDataSubsample.times.size)
                        for condition1 in conditions for condition2 in conditions}
        # Calculate all mahalanobis distances
        for t in range(len(covariances)):
            # Invert covariance matrix
            tInverseCovariance = numpy.linalg.inv(covariances[t])
            for condition1 in conditions:
                for condition2 in conditions:
                    # Calculate the difference between condition means
                    meanDifference = eval(f"erp_{condition1}[:,t]-erp_{condition2}[:,t]")
                    # Calculate the mahalanobis distance
                    mahaDistance = mahalanobis(meanDifference,numpy.zeros_like(meanDifference),tInverseCovariance)
                    # Store the mahalanobis distance
                    mahaDistances[f"{condition1}-{condition2}"][t] = mahaDistance
        
        # Initialize decoding
        decoding = numpy.zeros(mahaDistances[f"{conditions[0]}-{conditions[0]}"].shape)
        # Calculate decoding from mahalanobis distances

        for condition1 in conditions.tolist():
            for condition2 in conditions.tolist():
                # Add the distances between conditions
                if [condition1,condition2] in closeDistances:
                    decoding -= mahaDistances[f"{condition1}-{condition2}"]/(2*len(closeDistances))
                # Subtract the distances between same-condition trials
                if [condition1,condition2] in farDistances:
                    decoding += mahaDistances[f"{condition1}-{condition2}"]/(2*len(farDistances))
                    
        globalMahaDistances.append(mahaDistances)
        globalDecoding.append(decoding)
    
    return globalMahaDistances,globalDecoding

def mahaDistanceRSA(eegData,behavData,variableToSplit,splitType,subsamples=1,batchSize=1,nJobs=-1,
                    subsamplesSeed=1):
    # Set rng seed for control/replication
    rng = numpy.random.default_rng(subsamplesSeed)

    # Calculate the smallest sample size per condition
    variableCounts = behavData.value_counts(variableToSplit)
    weakestSampleSize = numpy.min(variableCounts)

    # Calculate the number of batches to be executed
    batches = subsamples//batchSize
    print("Batches: ",batches," size: ",batchSize)

    results = Parallel(n_jobs=nJobs,verbose=10)(delayed(_mahaDistanceRSA)(eegData,behavData,weakestSampleSize,
                                                                          i*batchSize,(i+1)*batchSize,variableToSplit,
                                                                          splitType,subsamplesSeed) for i in range(batches))
    
    globalMahaDistances, globalDecoding = zip(*results)
    globalDecoding = numpy.concatenate([i for i in globalDecoding])
    globalMahaDistances = [i for j in globalMahaDistances for i in j]

    return globalMahaDistances,globalDecoding

def _regressionRSA(eegData,behavData,weakestSampleSize,startSample,endSample,variablesToSplit,
                   similarityMatrices,subsampleSeed=1):
    from scipy.spatial.distance import squareform
    from scipy.spatial.distance import pdist
    from scipy.stats import zscore
    from sklearn.linear_model import LinearRegression
    import itertools
    from sklearn.covariance import LedoitWolf

    # Initialize decoding outputs
    globalRegressors = []

    # Create similarity matrices for regression
    similarityForRegression = numpy.array([squareform(matrix,checks=False) for matrix in similarityMatrices])
    similarityForRegression = zscore(similarityForRegression,axis=1).transpose()

    for subsample in range(startSample,endSample):
        print(f"Processing subsample {subsample}")
        # Set rng seed for control/replication
        rng = numpy.random.default_rng(subsampleSeed+subsample)        
        # Get subsample indices
        subsampleIndices = behavData.groupby(list(variablesToSplit.keys())).sample(weakestSampleSize,random_state=rng).index
        # Create the subsample
        behavDataSubsample = behavData.loc[subsampleIndices,:].copy().reset_index()
        eegDataSubsample = eegData[subsampleIndices].copy()
        
        # Initialize condition array
        averagesPerCondition = numpy.zeros((similarityMatrices[0].shape[0],
                                            eegData._data.shape[1],eegData._data.shape[2]))
        
        # Calculate averages per condition
        for conditionIndex,conditionValues in enumerate(itertools.product(*variablesToSplit.values())):
            mask = True
            for key,value in zip(variablesToSplit.keys(),conditionValues):
                mask &= behavDataSubsample[key]==value
            
            selectedDataEeg = eegDataSubsample._data[mask,:,:]
            averagesPerCondition[conditionIndex,:,:] = numpy.mean(selectedDataEeg,axis=0)

        globalRegressors.append([])
        # Perform regression
        for t in range(averagesPerCondition.shape[2]):
            currentTimeData = averagesPerCondition[:,:,t]
            # Calculate inverted covariance
            LW = LedoitWolf()
            tCovariance = LW.fit(eegDataSubsample._data[:,:,t]).covariance_
            tInverseCovariance = numpy.linalg.inv(tCovariance)
            # Calculate mahalanobis distance
            stimDistances = pdist(currentTimeData,metric='mahalanobis',VI=tInverseCovariance)
            stimDistances = zscore(stimDistances,ddof=1)
            # Perform regression
            currentTimeModel = LinearRegression()
            currentTimeModel.fit(similarityForRegression,stimDistances)
            # Store coefficients
            coefficientsToStore = numpy.append(numpy.array([currentTimeModel.intercept_]),currentTimeModel.coef_)
            globalRegressors[-1].append(coefficientsToStore)

    return globalRegressors

def regressionRSA(eegData,behavData,similarityMatrices,variablesToSplit,subsamples=1,batchSize=1,nJobs=-1,
                  subsamplesSeed=1):
    # Set rng seed for control/replication
    rng = numpy.random.default_rng(subsamplesSeed)

    # Calculate the smallest sample size per condition
    variableCounts = behavData.groupby(list(variablesToSplit.keys())).size()
    weakestSampleSize = numpy.min(variableCounts)

    # Calculate the number of batches to be executed
    batches = subsamples//batchSize
    print("Batches: ",batches," size: ",batchSize)

    results = Parallel(n_jobs=nJobs,verbose=10)(delayed(_regressionRSA)(eegData,behavData,weakestSampleSize,
                                                                        i*batchSize,(i+1)*batchSize,
                                                                        variablesToSplit,similarityMatrices,
                                                                        subsamplesSeed) for i in range(batches))
    
    globalRegressors = numpy.concatenate([i for i in results])

    return globalRegressors

def _slidingWindowRegressionRSA(eegData,behavData,weakestSampleSize,startSample,endSample,variablesToSplit,
                                similarityMatrices,slidingWindowSize,subsampleSeed=1):
    from scipy.spatial.distance import squareform
    from scipy.spatial.distance import pdist
    from scipy.stats import zscore
    from sklearn.linear_model import LinearRegression
    import itertools
    from sklearn.covariance import LedoitWolf

    # Initialize decoding outputs
    globalRegressors = []

    # Create similarity matrices for regression
    similarityForRegression = numpy.array([squareform(matrix,checks=False) for matrix in similarityMatrices])
    similarityForRegression = zscore(similarityForRegression,axis=1).transpose()

    for subsample in range(startSample,endSample):
        print(f"Processing subsample {subsample}")
        # Set rng seed for control/replication
        rng = numpy.random.default_rng(subsampleSeed+subsample)        
        # Get subsample indices
        subsampleIndices = behavData.groupby(list(variablesToSplit.keys())).sample(weakestSampleSize,random_state=rng).index
        # Create the subsample
        behavDataSubsample = behavData.loc[subsampleIndices,:].copy().reset_index()
        eegDataSubsample = eegData[subsampleIndices].copy()
        
        # Initialize condition array
        averagesPerCondition = numpy.zeros((similarityMatrices[0].shape[0],
                                            eegData._data.shape[1],eegData._data.shape[2]))
        
        # Calculate averages per condition
        for conditionIndex,conditionValues in enumerate(itertools.product(*variablesToSplit.values())):
            mask = True
            for key,value in zip(variablesToSplit.keys(),conditionValues):
                mask &= behavDataSubsample[key]==value
            
            selectedDataEeg = eegDataSubsample._data[mask,:,:]
            averagesPerCondition[conditionIndex,:,:] = numpy.mean(selectedDataEeg,axis=0)

        globalRegressors.append(numpy.zeros((eegDataSubsample._data.shape[2],similarityForRegression.shape[1]+1)))
        # Perform regression
        for t in range(slidingWindowSize,averagesPerCondition.shape[2]+1):
            currentTimeData = averagesPerCondition[:,:,t-slidingWindowSize:t].reshape(averagesPerCondition.shape[0],-1)
            # Calculate inverted covariance
            LW = LedoitWolf()
            tCovariance = LW.fit(eegDataSubsample._data[:,:,t-slidingWindowSize:t].reshape(eegDataSubsample._data.shape[0],-1)).covariance_
            tInverseCovariance = numpy.linalg.inv(tCovariance)
            # Calculate mahalanobis distance
            stimDistances = pdist(currentTimeData,metric='mahalanobis',VI=tInverseCovariance)
            stimDistances = zscore(stimDistances,ddof=1)
            # Perform regression
            currentTimeModel = LinearRegression()
            currentTimeModel.fit(similarityForRegression,stimDistances)
            # Store coefficients
            coefficientsToStore = numpy.append(numpy.array([currentTimeModel.intercept_]),currentTimeModel.coef_)
            globalRegressors[-1][t-1,:] = coefficientsToStore

    return globalRegressors

def slidingWindowRegressionRSA(eegData,behavData,similarityMatrices,variablesToSplit,slidingWindowSize=1,subsamples=1,
                               batchSize=1,nJobs=-1,subsamplesSeed=1):
    # Set rng seed for control/replication
    rng = numpy.random.default_rng(subsamplesSeed)

    # Calculate the smallest sample size per condition
    variableCounts = behavData.groupby(list(variablesToSplit.keys())).size()
    weakestSampleSize = numpy.min(variableCounts)

    # Calculate the number of batches to be executed
    batches = subsamples//batchSize
    print("Batches: ",batches," size: ",batchSize)

    results = Parallel(n_jobs=nJobs,verbose=10)(delayed(_slidingWindowRegressionRSA)(eegData,behavData,weakestSampleSize,
                                                                                     i*batchSize,(i+1)*batchSize,
                                                                                     variablesToSplit,similarityMatrices,
                                                                                     slidingWindowSize,
                                                                                     subsamplesSeed) for i in range(batches))
    
    globalRegressors = numpy.concatenate([i for i in results])

    return globalRegressors