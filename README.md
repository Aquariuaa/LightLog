# LightLog

## Introduction

LightLog is an open source deep learning based lightweight log analysis tool for log anomaly detection.

## Function description
[BGL&HDFS dataset and Methods of data processing] is for the processing of time-series data
- The BGL contains the complete steps for building word2vec models from structured logs
- Template saved as a .josn file
- Although rarely mentioned in our forthcoming paper, the treatment of the time-series data is very different from other work. 
The experimental results prove that this approach performs very well
[Enhanced TCN for Log Anomaly Detection on the BGL Dataset]
Validation of our method on the BGL dataset
[Enhanced TCN for Log Anomaly Detection on the HDFS Dataset]
Validation of our method on the HDFS dataset

##Note: 
1. This work includes the processing of BGL, HDFS datasets, training and testing of models, including details of building word2vec templates, PCA-PPA dimensionality reduction process and improved TCN
2. This work does not include log parsing，if you need to use it, please check [logparser](https://github.com/logpai/logparser)*
3. We highly recommend some other open source work as a complement and comparison to our work.[logdeep](https://github.com/donglee-afar/logdeep)

## Requirement

- python = 3.6
- tensorflow = 1.8.0
- Keras = 2.1.6
  
## Dataset
*Note: 
1. The data for the available experiments have been provided in our project.
2. The original dataset was too large for us to upload. Therefore, we have shared some links for you to download.
   [BGL](https://www.kaggle.com/omduggineni/loghub-bgl-log-data) and Special thanks for this research work
   [HDFS](https://github.com/donglee-afar/logdeep/tree/master/data/hdfs)
   
