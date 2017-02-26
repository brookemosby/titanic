import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import VotingClassifier as VC
from sklearn.linear_model import LogisticRegression as LR

def Read_Files(FileName):
    """
    Accepts file name, reads in file, converts and returns pd.DataFrame type
    """
    DataFrame=pd.read_csv(FileName)
    return DataFrame


def Changing_Name_to_Titles(DataFrame):
    """
    Accepts pd.DataFrame and takes from column name and creates column title filled with numbers corresponding to title type
    """
    DataFrame= Read_Files(DataFrame)
    titles=DataFrame['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])
    title_mapping = {"the":5, "Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Dr.": 5, "Rev.": 6, "Major.": 7, "Col.": 7, "Mlle.": 2, "Mme.": 3, "Don.": 9, "Lady.": 10, "Countess.": 10, "Jonkheer.": 10, "Sir.": 9, "Capt.": 7, "Ms.": 2, "Dona.": 10}
    for k,v in title_mapping.items():
        titles[titles == k] = v
    DataFrame["Title"] = titles
    return DataFrame


def Creating_NewCols(DataFrame):
    """
    Accepts pd.DataFrame and creates columns NameLen- len of name string & FamSize- # of siblings addded to # of parents & children.
    Modifies Cabin column so that the cabin letter is mapped to a number corresponding to the cabin.
    Deletes columns 'Parch', 'SibSp', and 'PassengerId', and then returns DataFrame
    """
    DataFrame= Changing_Name_to_Titles(DataFrame)
    DataFrame['NameLen']=DataFrame['Name'].apply(lambda x: len(x))
    DataFrame['FamSize']=DataFrame['SibSp']+DataFrame['Parch']
    cabins=DataFrame['Cabin'].apply(lambda x:   str(x)[0])
    cabin_mapping={'A':3,'B':5,'C':5,'D':4,'E':4,'F':3,'G':2,'T':1,'n':10}
    for k,v in cabin_mapping.items():
        cabins[cabins==k]=v
    DataFrame['Cabin']=cabins
    del DataFrame['Parch']
    del DataFrame['SibSp']
    del DataFrame['PassengerId']
    return DataFrame


def Getting_Dummies(DataFrame,train):
    """
    Accepts DataFrame, and training set DataFrame, Creates dummmies for columns 'Pclass', 'Embarked', and 'Sex'.
    Then deletes previous columns, along with 'Ticket', and fills in missing values for 'Age' and 'Fare'.
    Returns DataFrame
    """
    
    DataFrame=Creating_NewCols(DataFrame)
    pclass = pd.get_dummies( DataFrame.Pclass , prefix='Pclass' )
    sex = pd.get_dummies(DataFrame.Sex)
    embarked = pd.get_dummies(DataFrame.Embarked, prefix='Embarked')
    DataFrame=pd.concat([DataFrame,pclass,sex,embarked],axis=1)
    del DataFrame['Pclass']
    del DataFrame['Name']
    del DataFrame['Sex']
    del DataFrame['Ticket']
    del DataFrame['Embarked']
    DataFrame['Fare'].fillna(train['Fare'].median(), inplace = True)
    DataFrame['Age'].fillna(train['Age'].median(), inplace = True)
    return DataFrame


def Create_Decision_Tree(train='data/train.csv'):
    """
    Creates Decision Tree fitted with default train='data/train.csv', returns Decision Tree
    ~75%
    """
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    DT = DTC(random_state=99)
    DT.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return DT

def Create_Random_Forest(train='data/train.csv'):
    """
    Creates and returns Random Forest fitted with default parameter train='data/train.csv'
    ~77% when there is 100 n_estimators
    ~78.4% when there is 300 n_estimators
    """
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    RF = RFC(min_samples_split=10, n_estimators= 700, criterion= 'gini', max_depth=None)
    RF.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return RF

def Create_KNeighbors(train='data/train.csv'):
    """
    Creates KNeighbors model fitted with default train='data/train.csv', returns KNeighbors model.
    ~65%
    """
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    KN = KNC(n_neighbors=3)
    KN.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return KN

def Create_Logistic_Regression(train='data/train.csv'):
    """
    ~76%
    """
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    LogR= LR()
    LogR.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return LogR


def Create_Voting(train='data/train.csv'):
    """
    Creates and returns a model based off of the weighted classifiers Decision Tree, Random Forest and KNeighbors
    ~76.5%- voting is soft and weights are [6,8,7]
    """
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    Clf1=Create_Decision_Tree()
    Clf2=Create_Random_Forest()
    Clf3=Create_Logistic_Regression()
    Clf = VC(estimators=[('dt', Clf1), ('rf', Clf2), ('lr', Clf3)], voting='hard')
    Clf.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return Clf
    

def Produce_Predictions(FileName,train='data/train.csv',test='data/test.csv'):
    """
    Accepts file name, 'example.csv', to print predictions too, along with default train='data/train.csv' & test='data/test.csv'.
    Uses Decision Tree to create predictions on who survived.
    returns nothing, creates 'example.csv'
    """
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    test=Getting_Dummies(test,trainDF)
    MLA=Create_Voting()
    predictions = MLA.predict(test)
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    test = pd.read_csv( 'data/test.csv')
    predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
    predictions.to_csv(FileName, sep=",", index = False)
    #~ 75% :)
    
    
if __name__=="__main__":
    Produce_Predictions('TestRun.csv')
