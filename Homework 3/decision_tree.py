import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y,0)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        #x: sample * feature
        #y: true labels
        sample_num, feature_num = X.shape
        class_num = len(np.unique(y))
        
        #stopping condition
        if(depth>=self.max_depth or class_num==1 or sample_num<2):
            leaf_val = np.argmax(np.bincount(y))#找到y中出現最多次的class
            self.progress.update(1)#update 進度條
            return{'type': 'leaf', 'value':leaf_val}

        idx, thres = self._best_split(X,y)
        #no split is found, create a leaf node
        if idx is None:
            val = np.argmax(np.bincount(y))
            self.progress.update(1)
            return {'type': 'leaf', 'value': val}
        
        xl, yl, xr, yr = self._split_data(X,y,idx, thres)
        if len(xr)==0 or len(xl)==0:
            val = np.argmax(np.bincount(y))
            self.progress.update(1)
            return {'type': 'leaf', 'value': val}

        left_subtree = self._build_tree(xl, yl, depth+1)
        right_subtree = self._build_tree(xr, yr, depth+1)
        
        return {
            'type': 'node',
            'feature_idx': idx,
            'threshold': thres,
            'left': left_subtree,
            'right': right_subtree
        }
        

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        predictions=[]
        for i in range(len(X)):
            x=X.iloc[i]#for each sample in X
            predict=self._predict_tree(x, self.tree)
            predictions.append(predict)
        return torch.tensor(predictions)

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        if tree_node['type']=='leaf':
            return tree_node['value']
        
        feature_idx=tree_node['feature_idx']
        threshold=tree_node['threshold']
        feature_val=x[feature_idx]
        
        if feature_val<=threshold:
            return self._predict_tree(x,tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        mask = X.iloc[:, feature_index] <= threshold
        mask_np=np.array(mask)#convert from pandas into numpy
        y=np.array(y)
        
        left_X = X[mask]
        left_y = y[mask_np]
        
        right_X = X[~mask]
        right_y = y[~mask_np]
        
        return left_X, left_y, right_X, right_y

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_threshold=None
        best_feature_index=None
        best_infogain=-float('inf')
        current_entropy=self._entropy(y)
        sample_num=len(y)
        
        if X.shape[1]>100:
            used_feature=np.random.choice(X.shape[1],100,replace=False)
        else:
            used_feature = range(X.shape[1])
        
        for feature_idx in used_feature:
            #try each feature, find the best split
            feature_values = sorted(X.iloc[:, feature_idx].unique())
            #取出feature_idx那一列並排序去重
            if len(feature_values)<=1:
                continue
            if len(feature_values)>10:
                thresholds = np.percentile(feature_values, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            else:
                thresholds = [(feature_values[i]+feature_values[i+1])/2 for i in range(len(feature_values)-1)]
            for thres in thresholds:
                xl, yl, xr, yr = self._split_data(X,y,feature_idx, thres)
                if len(yl)==0 or len(yr)==0:#doesn't split the data
                    continue
                lweight = len(yl)/sample_num
                rweight = len(yr)/sample_num
                lentropy=self._entropy(yl)
                rentropy = self._entropy(yr)
                new_entropy = lweight*lentropy+rweight*rentropy
                infogain = current_entropy-new_entropy
                if infogain>best_infogain:
                    best_infogain=infogain
                    best_feature_index=feature_idx
                    best_threshold=thres
        return best_feature_index, best_threshold

    def _entropy(self, y: np.ndarray)->float:
        # (TODO) Return the entropy
        if len(y)==0:
            return 0
        _, counts=np.unique(y,return_counts=True)
        p=counts/len(y)#probability of each class
        entropy=-np.sum(p*np.log2(p+1e-10))
        return entropy
    

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    model.eval()#evaluation mode
    features = []
    labels = []
    
    with torch.no_grad():#don't track gradient
        for images, label in tqdm(dataloader, desc = 'extracting features and labels'):
            images = images.to(device)
            feature = model(images)
            features.extend(feature.cpu().numpy())
            labels.extend(label.cpu().numpy())
            
    features = pd.DataFrame(features)
    
    return features, labels

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    model.eval()#evaluation mode
    features = []
    paths = []
    with torch.no_grad():
        for images, path in tqdm(dataloader, desc = 'extracting features and paths'):
            images = images.to(device)
            feature = model(images)
            features.extend(feature.cpu().numpy())
            paths.extend(path)
    features = pd.DataFrame(features)
    return features, paths