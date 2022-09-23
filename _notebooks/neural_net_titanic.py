import pandas as pd

# uncomment and import modules
# pip install fastbook
# import fastbook
# fastbook.setup_book()
# from fastai.vision.all import *
# from fastbook import *
# import torch.nn.functional as F

# to run from your workstation
# download the titanic survival data set : train.csv
import os
path = os.getcwd()
# df = pd.read_csv(f"{path}/YOUR_FILE_LOCATION/train.csv")

df = pd.read_csv('/Users/ericvincent/Desktop/train.csv')

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

def sigmoid(x): return 1/(1+torch.exp(-x))

def survive_loss_updated(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

class FormatDataframe :
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def splitData(self):
        twenty_percent_df = df
        twenty_percent_df['Male'] = twenty_percent_df['Sex']
        twenty_percent_df['Male'] = twenty_percent_df['Male'].replace({'male': 1, 'female' : 0})
        twenty_percent_df['Embarked_C'] = twenty_percent_df['Embarked']
        twenty_percent_df['Embarked_C'] = twenty_percent_df['Embarked_C'].replace({'S':0, 'C':1, 'Q':0})
        twenty_percent_df['Embarked_S'] = twenty_percent_df['Embarked']
        twenty_percent_df['Embarked_S'] = twenty_percent_df['Embarked_S'].replace({'S':1, 'C':0, 'Q':0})
        twenty_percent_df['Pclass1'] = twenty_percent_df['Pclass']
        twenty_percent_df['Pclass2'] = twenty_percent_df['Pclass']
        twenty_percent_df['Pclass1'] = twenty_percent_df['Pclass1'].replace({2:0, 3:0})
        twenty_percent_df['Pclass2'] = twenty_percent_df['Pclass2'].replace({1:0, 3:0, 2:1})
        twenty_percent_df = twenty_percent_df.drop(columns=['Sex', 'Age', 'Fare', 'Embarked', 'Pclass'])
        twenty_percent_df['Embarked_S'] = twenty_percent_df['Embarked_S'].fillna(0)
        twenty_percent_df['Embarked_C'] = twenty_percent_df['Embarked_C'].fillna(0)
        twenty_percent_df = twenty_percent_df.drop(columns=['Name', 'Cabin', 'PassengerId', 'Ticket'])
        eighty_percent_df = twenty_percent_df.iloc[180:]
        twenty_percent_df = twenty_percent_df.iloc[:179]

        return eighty_percent_df, twenty_percent_df
    
    def createTensors(self, dfName):
        # return labels
        survived_label_train = dfName['Survived'] == 1
        death_label_train = dfName['Survived'] == 0
        
        # creating tensors
        survived_df = dfName.loc[survived_label_train]
        death_df = dfName.loc[death_label_train]

        stacked_survived = [tensor(survived_df.iloc[num]) for num in range(len(survived_df))]
        stacked_death = [tensor(death_df.iloc[num]) for num in range(len(death_df))]
        
        survive_tensors_stacked = torch.stack(stacked_survived).float()
        death_tensors_stacked = torch.stack(stacked_death).float()

        return survive_tensors_stacked, death_tensors_stacked



if __name__ == "__main__" :

    model = FormatDataframe(df)
    train, validation = model.splitData() 
    survive_tensors_stacked_train, death_tensors_stacked_train = model.createTensors(train)
    survive_tensors_stacked_validation, death_tensors_stacked_validation = model.createTensors(validation)
    
    # labels on 80% of data
    label_df = FormatDataframe(df)
    eighty_percent_labels,_ = label_df.splitData()
    survived_label = eighty_percent_labels['Survived'] == 1
    death_label = eighty_percent_labels['Survived'] == 0
    survived = eighty_percent_labels.loc[survived_label]
    death = eighty_percent_labels.loc[death_label]

    # create training dl
    train_x = torch.cat([survive_tensors_stacked_train, death_tensors_stacked_train]).view(-1, 8)
    train_y = tensor([1]*len(survived) + [0]*len(death)).unsqueeze(1)
    dset = list(zip(train_x,train_y))
    dl = DataLoader(dset, batch_size=8)
    
    # create validation dl
    valid_x = torch.cat([survive_tensors_stacked_validation, death_tensors_stacked_validation]).view(-1, 8)
    valid_y = tensor([1]*len(survive_tensors_stacked_validation) + [0]*len(death_tensors_stacked_validation)).unsqueeze(1)
    valid_dset = list(zip(valid_x,valid_y))
    valid_dl = DataLoader(valid_dset, batch_size=8)
    
    # finally
    dls = DataLoaders(dl, valid_dl)
    
    # neural net
    simple_net = nn.Sequential(
        nn.Linear(8,1),
        nn.ReLU(),
        nn.Linear(1,8)
    )
    
    learn = Learner(dls, simple_net, opt_func=SGD,
                    loss_func=survive_loss_updated, metrics=batch_accuracy)
    
    learn.fit(40, 0.1)