import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC

def Feature_Engineering(DataFrame,train):
    """
    Extracts important features and writes them in usable form
    Deletes features of little importance
    
    :param DataFrame: This is the file name of a csv file we wish to convert into a usable DataFrame.
    :param train: This is training set corresponding to our csv file. Should be of type pandas.DataFrame
    :returns: Returns csv file, after having been modified as a pandas.DataFrame type
    """
    
    
    DataFrame= pd.read_csv(DataFrame)
    titles=DataFrame['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])
    title_mapping = {"the":5, "Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Dr.": 5, "Rev.": 6, "Major.": 7, "Col.": 7, "Mlle.": 2, "Mme.": 3, "Don.": 9, "Lady.": 10, "Countess.": 10, "Jonkheer.": 10, "Sir.": 9, "Capt.": 7, "Ms.": 2, "Dona.": 10}
    for k,v in title_mapping.items():
        titles[titles == k] = v
    DataFrame["Title"] = titles

    DataFrame['NameLen']=DataFrame['Name'].apply(lambda x: len(x))
    DataFrame['FamSize']=DataFrame['SibSp']+DataFrame['Parch']
    DataFrame['Has_Cabin'] = DataFrame["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    cabins=DataFrame['Cabin'].apply(lambda x:   str(x)[0])
    cabin_mapping={'A':3,'B':5,'C':5,'D':4,'E':4,'F':3,'G':2,'T':1,'n':10}
    for k,v in cabin_mapping.items():
        cabins[cabins==k]=v
    DataFrame['Cabin']=cabins  
    del DataFrame['Parch']
    del DataFrame['SibSp']
    del DataFrame['PassengerId']

    pclass = pd.get_dummies( DataFrame.Pclass , prefix='Pclass' )
    sex = pd.get_dummies(DataFrame.Sex)
    embarked = pd.get_dummies(DataFrame.Embarked, prefix='Embarked')
    DataFrame=pd.concat([DataFrame,pclass,sex,embarked],axis=1)
    del DataFrame['Pclass']
    del DataFrame['Name']
    del DataFrame['Ticket']
    del DataFrame['Sex']
    del DataFrame['Embarked']
    
    DataFrame['Fare'].fillna(train['Fare'].median(), inplace = True)
        # Mapping Fare
    DataFrame.loc[ DataFrame['Fare'] <= 7.91, 'Fare'] 						        = 0
    DataFrame.loc[(DataFrame['Fare'] > 7.91) & (DataFrame['Fare'] <= 14.454), 'Fare'] = 1
    DataFrame.loc[(DataFrame['Fare'] > 14.454) & (DataFrame['Fare'] <= 31), 'Fare']   = 2
    DataFrame.loc[ DataFrame['Fare'] > 31, 'Fare'] 							        = 3
    DataFrame['Fare'] = DataFrame['Fare'].astype(int)
    DataFrame['Age'].fillna(train['Age'].median(), inplace = True)

    return DataFrame


def Create_Random_Forest(train):
    """
    Fits Random Forest to training set.
    
    :param train: This is the file name of a csv file we wish to have fitted to a Random Forest, does not need to have features already extracted.
    :returns: Returns sklearn.ensemble.Random_Forest_Classifier fitted to training set.
    """
    trainDF=pd.read_csv(train)
    train=Feature_Engineering(train,trainDF)
    RF = RFC(min_samples_split=10, n_estimators= 700, criterion= 'gini', max_depth=None)
    RF.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return RF


def Produce_Predictions(FileName,train,test):
    """
    Produces predictions for testing set, based off of training set.
    
    :param FileName: This is the csv file name we wish to have our predictions exported to.
    :param train: This is the file name of a csv file that will be the training set.
    :param test: This is the file name of the testing set that predictions will be made for.
    :returns: Returns nothing, creates csv file containing predictions for testing set.
    """
    TestFileName=test
    TrainFileName=train
    trainDF=pd.read_csv(train)
    train=Feature_Engineering(train,trainDF)
    test=Feature_Engineering(test,trainDF)
    MLA=Create_Random_Forest(TrainFileName)
    predictions = MLA.predict(test)
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    test = pd.read_csv(TestFileName)
    predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
    predictions.to_csv(FileName, sep=",", index = False)
    
if __name__=="__main__":
    Produce_Predictions('TestRun.csv') # pragma: no cover
