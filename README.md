Covid_track
Pytorch implementation of the DenseNet121 network trained on CheXpert dataset to extract features from pulmonary X-Ray for COVID risk assessment

This project contains the code to reproduce the main results from the article 
'Tracking and predicting COVID-19 radiological trajectory on chest X-rays using deep learning.'
We provide the features extracted from the COVID-19 image data repository (link to the source images available in supplemental materials)
We provide the features extracted from multi-institutional ICU dataset codeair (source images unavailable)
The trained deep learning feature extractor is available and the code to train it is available.
I preventively apologize for the quality of the code, I was not aware of the pipelining features of scikitlearn at the time of doing this work.

Please use the dockerfile to avoid compatibility issues. The version of pytorch used is 1.4.
The radiological_assessment.py file is supposed to reproduce all the results from the article in terms of radiological trajectory prediction.
It also contains the results for the radiological severity assessment. 

The model_training.py file can be used to retrain a feature extractor or extract features/radiological signs from images. It will require some modifications
to fit with your filesystem and the location of the images you want to train on or extract features from.