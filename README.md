# Heart-Disease
Heart Disease ML Project 


This project focused on applying a scikit-learn algorithm to data on heart disease patients in order to detect the presence of heart disease. 
The goal of the project was to have an algorithm accurately predict if a patient had heart disease or not, based on 303 data samples of 14 attributes from a Cleveland hospital. The dataset was from UC Irvine, and was modified to remove samples that contained NaN entries, and all 4 values that indicated heart disease were tranformed to 1 value to give a 0-1 prediction. The program uses a variety of sklearn and numpy libraries. 

The program identified a 0 or 1 value for the presence of heart disease over 149 test samples. it had a precision and recall both of 0.83 and an f1-score of 0.82. Modifying the ratio of training to testing data from the current 50/50 has a negative impact of each of those scores. Adding more data  to train and test with could benefit the program or change how it reacts to different ratios, but other international data from the same dataset is incomplete and wasn't used. 

Replicating these results can be done with sciki-learn and the dataset linked. 

This programs could be improved by using a different scikit algorithm, and the addition of a larger data set. If there was enough data, bringing back the scale of 0-4 predictions would be a worthy improvement. 





For Information on the dataset:


https://archive.ics.uci.edu/ml/datasets/heart+disease



The file, "heart-disease.names" contains the following:

    Publication Request
  
    Title
  
    Source Information
  
    Past Usage
  
    Relevant Information
  
    Number of Instances
  
    Number of Attributes
  
    Attribute Information
  
    Missing Attribute Values
  
    Class Distribution
  
