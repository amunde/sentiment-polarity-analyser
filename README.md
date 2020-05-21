# sentiment-polarity-analyser

This repository contains a sentiment polarity analyzer, using binary logistic regression from scratch in Python.
The implementation consists of two programs, a feature extraction program (feature.py) and a sentiment analyzer program (lr.py) using binary logistic regression. 

Feature.py - 

It has argument feature_flag that specifies the type of feature extraction to be used further by the logistic regression model. 
feature_flag = 1 => This model will use the dense representation of the input file.
feature_flag = 2 => This model will use the sparse representation of the input file.

Execution - 
python feature.py [args1...]

Where above [args1...] is a placeholder for eight command-line arguments described in detail below:
1. train input: path to the training input .tsv file 
2. validation input: path to the validation input .tsv file 
3. test input: path to the test input .tsv file 
4. dict input: path to the dictionary input .txt file 
5. formatted train out: path to output .tsv file to which the feature extractions on the train- ing data should be written 
6. formatted validation out: path to output .tsv file to which the feature extractions on the validation data should be written 
7. formatted test out: path to output .tsv file to which the feature extractions on the test data should be written
8. feature flag: integer taking value 1 or 2 that specifies whether to construct the Model 1 feature set or the Model 2 feature set

eg - python feature.py train_data.tsv valid_data.tsv test_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1


lr.py - 

The second program lr.py,  implements a sentiment polarity analyzer using binary logistic regression. The file learns the parameters of a binary logistic regression model that predicts a sentiment polarity (i.e. label) for the corresponding feature vector of input file.

Execution - 

python feature.py [args2...]

On the other hand, [args2...] is a placeholder for eight command-line arguments described in detail below:
1. formatted train input: path to the formatted training input .tsv file 
2. formatted validation input: path to the formatted validation input .tsv file 
3. formatted test input: path to the formatted test input .tsv file 
4. dict input: path to the dictionary input .txt file 
5. train out: path to output .labels file to which the prediction on the training data should be written
6. test out: path to output .labels file to which the prediction on the test data should be written 
7. metrics out: path of the output .txt file to which metrics such as train and test error should be written 
8. num epoch: integer specifying the number of times SGD loops through all of the training data 

eg - python lr.py formatted_train.tsv formatted_valid.tsv formatted_test .tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60
