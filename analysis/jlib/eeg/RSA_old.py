import numpy

def halfSplitMahaDistanceRSA_OLD(eegData,behavData,variableToSplit,splitType=None,subsamples=1,subsamplesSeed=1):
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import LedoitWolf

    # Set rng seed for control/replication
    rng = numpy.random.default_rng(subsamplesSeed)

    # Calculate the smallest sample size per condition
    variableCounts = behavData.value_counts(variableToSplit)
    weakestSamplesize = numpy.min(variableCounts)

    # Initialize decoding outputs
    globalDecoding = []
    globalMahaDistances = []

    for subsample in range(subsamples):
        print(f"Processing subsample {subsample}")
        # Get subsample indices
        subsampleIndices = behavData.groupby(variableToSplit).sample(weakestSamplesize,random_state=rng).index
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