# irregulartimeseries

The notebooks are listed in the root of the project at the moment.

### 00-DataOverview
Contains basic visualization of the data.

### 00-ModelResults
Aggregates the model results across all notebooks and visualizes them.

### 01-Basic
Uses the data on all models to get a base performance metric.


### 03-Features
These notebooks try to provide different features as input to the models.

> n stands for normalized feature input vectors
> p stands for appending the predictor value into the feature vector.

#### 03-FeaturesDate
Uses <YYYY, MM, DD> as the feature vector.

#### 03-FeaturesOther
Uses <actual_min_temp, actual_max_temp, actual_precipitation> as the feature vector.



### 04-Remove
These notebooks completely remove time points from the data.  The number at the end of the title represents the percent of data that will be removed. 


### 05-Missing
These notebooks mark time points as being missing.