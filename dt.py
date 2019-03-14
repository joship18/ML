import numpy as np
import pandas as pd
df = pd.read_csv("iris.csv")
#df.head()
from pandas.api.types import is_numeric_dtype
for i in range(df.shape[1]):
    if(is_numeric_dtype(df.iloc[:,i])):
        mask = df.iloc[:, i] < df.iloc[:, i].median()
        df.ix[mask, i] = 0
        df.ix[~mask, i] = 1
        
#df.head()
df = df.dropna()
mask = np.random.rand(len(df)) < 0.8
train = df.loc[mask]
test = df.loc[~mask]

def entropy(vec):
    m = len(vec)
    E = vec.value_counts()
    n = len(E)
    E = E/m
    E = - E*np.log2(E)
    return (E.sum() / np.log2(n))
    
    
def find_best_split(df):
    best_gain = 0
    best_attribute = None
    
    current_entropy = entropy(df.iloc[:,-1])
    
    n_features = df.shape[1] - 1
    
    if(n_features == 0):
        return best_gain, best_attribute
    
    n_rows = df.shape[0]
    
    for i in range(n_features):
        entropy_after_split = 0
        attribute_values = df.iloc[:,i].unique()
        for j in attribute_values:
            mask = (df.iloc[:,i] == j)
            entropy_after_split = entropy_after_split + (len(mask[mask == True]) / n_rows )* entropy(df.ix[mask,-1])
        gain = current_entropy - entropy_after_split
        if(gain > best_gain):
            best_gain = gain
            best_attribute = i
    
    return best_gain, best_attribute
  
  
def partition(df, attribute):
    
    
    partitioned_data = {}
    
    attribute_values = df.iloc[:,attribute].unique()
    
    for i in attribute_values:
        mask = (df.iloc[:,attribute] == i)
        partitioned_data[i] = (df.loc[mask, :])
    
    for i in range(len(partitioned_data)):
        partitioned_data[i].drop(list(df)[attribute], axis=1) 
    
    return partitioned_data


class Leaf_node:
    def __init__(self, df):
        self.prediction = {}
        X = df.iloc[:, -1].mode()
        n = len(X)
        for i in X:
            self.prediction[i] = 100/n
            

class Decision_node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
    def add_child(self, df, value):
        self.children[value] = df
        
        
def build_tree(df):
    gain, attribute = find_best_split(df)
    
    if(gain == 0):
        return Leaf_node(df)
    
    partitioned_data = partition(df, attribute)
    
    X = Decision_node(list(df)[attribute])
    
    for i in range(len(partitioned_data)):
        X.add_child(build_tree(partitioned_data[i]), i)
    
    return X



head = build_tree(train)


def predict(test, head):
    correct_predictions = 0
    for i in test.index:
        node = head
        while(isinstance(node, Leaf_node) == False):
            Z = test.loc[i,node.attribute]
            node = node.children[Z]
        print("actual class: {} \t predicted class: {}" .format(test.loc[i, "species"], node.prediction))
        if(test.loc[i, "species"] in  node.prediction):
            correct_predictions += 1
    print("\naccuracy = {}%\n" .format(correct_predictions*100/len(test)))
    

predict(test, head)
