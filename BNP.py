#
#	Kaggle - BNP Paribas Cardif Claims Management
#	Summary: Priditict the probability of the the target column value by Logistic Regression Algorithm
#	Platform: Window 7(64 bits), Python 2.7(64 bits)
#


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets


# Prepare the training and testing dataset
dfTrain = pd.read_csv('../datasets/BNP/train.csv')
dfTest = pd.read_csv('../datasets/BNP/test.csv')

dfTest.insert(loc = 2, column = 'target', value = 0.5)


num_Feature = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 
			'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 
			'v21', 'v23', 'v25', 'v26' ,'v27', 'v28', 'v29', 'v32', 'v33',
			'v34', 'v35', 'v36', 'v37', 'v39', 'v40', 'v41', 'v42', 'v43', 
			'v44', 'v45', 'v46', 'v48', 'v49','v50', 'v51', 'v53', 'v54',
			'v55', 'v57', 'v58', 'v59',	'v60', 'v61', 'v63', 'v64', 'v65', 
			'v67', 'v68', 'v69', 'v70', 'v73', 'v76', 'v77', 'v78', 'v80',
			'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88', 'v89',
			'v90', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99',
			'v100', 'v101', 'v102', 'v103', 'v104', 'v105', 'v106', 'v108',
			'v109', 'v111', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119',
			'v120', 'v121', 'v122', 'v123', 'v124', 'v126', 'v127', 'v128',
			'v130', 'v131']
ctg_Feature = ['v3', 'v24', 'v30', 'v31', 'v38', 'v47', 'v52', 'v62', 'v66', 'v71',
			'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v129']
st_Feature = ['ID', 'target']
id_Feature = ['ID']
# discptive numeric value columns: v38 and v62

dict = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10,
		'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18, 'S':19, 'T':20,
		'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}


num_Train_clean = dfTrain[num_Feature].fillna(dfTrain[num_Feature].mean())
ctg_Train_clean = dfTrain[ctg_Feature].fillna(method = 'ffill')
ctg_Train_clean = dfTrain[ctg_Feature].fillna(method = 'bfill')
ctg_Train_clean = ctg_Train_clean.replace(dict)

num_Test_clean = dfTest[num_Feature].fillna(dfTest[num_Feature].mean())
ctg_Test_clean = dfTest[ctg_Feature].fillna(method = 'ffill')
ctg_Test_clean = dfTest[ctg_Feature].fillna(method = 'bfill')
ctg_Test_clean = ctg_Test_clean.replace(dict)

# transform rank alpha to number.
train_Clean = num_Train_clean.join(ctg_Train_clean)
train_Clean = dfTrain[st_Feature].join(train_Clean)
train_Clean.round(5)

test_Clean = num_Test_clean.join(ctg_Test_clean)
test_Clean = dfTest[st_Feature].join(test_Clean)
test_Clean.round(5)

# Back up clean data
train_Clean.to_csv('../datasets/BNP/train_clean.csv')
test_Clean.to_csv('../datasets/BNP/test_clean.csv')


# Generate the Logistic model from the training dataset
features = num_Feature + ctg_Feature

trainSample = train_Clean[features]
trainTarget = train_Clean['target']

classifier = linear_model.LogisticRegression().fit(trainSample, trainTarget)


# Predict the probability of the testing dataset
testSample = test_Clean[features]
testTarget = test_Clean['target']

testTarget = classifier.predict(testSample)

resultId = test_Clean[id_Feature]
dfResult = resultId.join(pd.DataFrame(testTarget))

dfResult.rename(columns = {0:'PredictedProb'}, inplace = True)
dfResult.to_csv('../datasets/BNP/submission.csv', header = True)
print dfResult
