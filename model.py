import numpy as np
from tqdm import tqdm

class linearclassifier(object):

    def __init__(self, lr = 0.01, epochs = 1000) -> None:
        self.w = None               # Weight
        self.b = None               # Bisd
        self.lr = lr                # Learning Rate
        self.epochs = epochs        # Training Iteration
        self.history_loss = []      # Store Loss in list 

    
    # Access model parameters
    def parameters(self):
        params = {'Weight':self.w,
                  'Bias':self.b,
                  'Learning rate':self.lr,
                  'Epochs':self.epochs
                  }
        return params
    
    # Sigmoid Function
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    

    def predict(self, x):
        '''    
        Arg:
            x : input data

        Retuen:
            
        '''
        output = np.dot(x, self.w) + self.b
        y_pred = self._sigmoid(output)
        y_pred_classes = np.array([1 if i > 0.5 else 0 for i in y_pred]
)

        return y_pred_classes
    

    def fit(self, X, y):
        '''
        fit model by data

        Arg:        
            X : input data
            y : label

        Return:
            None

        '''
        n_data, n_features = X.shape

        # initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0

        #更新權重
        for i in range(self.epochs):
            
            #predict
            output = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(output)

            #Cross Entropy loss
            loss = -(1 / n_data * (np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))))
            self.history_loss.append(loss)

            #gradient descent
            dw = 1 / n_data * np.dot(X.T, (y_pred - y))
            db = 1 / n_data * np.sum(y_pred - y)


            # renow parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            print(f"Epoch[{i + 1}/{self.epochs}], Loss : {loss}")

##################################### KNN ####################################

class Node(object):
    def __init__(self, data, label, depth, rchild = None, lchild = None, ) -> None:
        self.right = rchild
        self.left = lchild
        self.depth = depth
        self.data = data
        self.label = label

    


class KDTree(object):

    def __init__(self, X, y, method = "'euclidean'"):
        y = y.reshape((-1,1))
        self.data = np.hstack([X, y])
        #物件實現時就會遞迴建立樹
        self.root = self.buildtree(self.data)
        self.method = method

    def buildtree(self, data, depth = 0):
        if len(data) <= 0:
            return
        
        #print(np.shape(data))
        
        n, self.p = np.shape(data)

        aim_axis = depth % (self.p - 1)

        sorted_data = sorted(data, key = lambda x : x[aim_axis])
        mid = n // 2         
        

        node = Node(sorted_data[mid][:-1], sorted_data[mid][-1], depth = depth)

        node.left = self.buildtree(sorted_data[:mid], depth = depth + 1)
        node.right = self.buildtree(sorted_data[mid + 1:], depth = depth + 1)

        return node
    
    def _distance(self, x1, x2):
        if self.method == 'euclidean':
            return np.linalg.norm(x1 - x2)
        
        elif self.method == 'manhattan':
            return np.sum(np.abs(x1, x2))
                
        elif self.method == 'cosin similarity':
            return 1 - (np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
        
        else:
            raise ValueError("Invalid distance metric")

        

    def search(self, x, k = 3):

        self.nearest = []
        assert  1 <= k and k <= len(self.data), "K is ou if range"

        def recurve(node):
            if node is None:
                return
            
            #當前用來切分資料的維度
            now_axis = node.depth % (self.p - 1)


            # print(f"search:{np.shape(x)}")
            # print(f"x[{now_axis}] : {x[now_axis]}")
            # print(f"node.data[{now_axis}] : {node.data[now_axis]}")


            #遞迴搜尋
            if (x[now_axis] < node.data[now_axis]):
                recurve(node.left)
            else:
                recurve(node.right)

            
            #搜尋結束，開始計算距離
            dist = self._distance(x, node.data)

            #如果還沒搜尋到k個最近則加入list
            #反之，則去除list中距離最遠的node
            if(len(self.nearest) < k):
                self.nearest.append([node, dist])
            else:
                for i, d in enumerate(self.nearest):
                    if (d[1] > dist):
                        self.nearest[i] = [node, dist]


            #檢查隔壁子空間是否有距離較近的點
            max_index = np.argmax(np.array(self.nearest)[:,1])

            if (self.nearest[max_index][1] > abs(x[now_axis] - node.data[now_axis])):
                if (x[now_axis] - node.data[now_axis]) < 0:
                    recurve(node.right)
                else:
                    recurve(node.left)


        recurve(self.root)

        y_pred = np.bincount([item[0].label for item in self.nearest]).argmax()

        return self.nearest, y_pred




class KNN(object):

    def __init__(self, K = 3, method = 'euclidean') :
        self.k = K
        self.method = method


    def parameters(self):
        params = {'K':self.k,
                  'Method':self.method,
                  }
        return params 

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.kdtree = KDTree(self.X, self.y, self.method)
        print("Fitting...")
    
    
    def predict(self, X):
        
        nearest, y_pred = self.kdtree.search(X, self.k)

        return nearest, y_pred

    def score(self,xtest, ytest):
        y_pred_list = []
        for x in tqdm(xtest):
            #print(f"score : {np.shape(x)}")
            _,  y_pred = self.predict(x)
            y_pred_list.append(y_pred)

        acc = np.sum(np.array(y_pred_list) == ytest) / len(ytest)

        print(f"The Accuracy of Testing Data : {acc :.2%}")



        
##################################### Decision Tree ####################################

class TreeNode(object):
    def __init__(self, feature_index = None, threshold = None, right = None, left = None, info_gain = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value




class DecisionTree(object):
    def __init__(self, max_depth = 10, min_sample_spilt = None) -> None:
        self.tree = None
        self.max_depth = max_depth
        self.min_sample_spilt = min_sample_spilt


    def parameters(self):

        params = {"Max_depth" : self.max_depth,
                  "Min_Sample_Split" : self.min_sample_spilt}
        
        return params
    


    def fit(self, X, y):
        
        self.tree = self._build_tree(X, y, depth = 0)

    def _build_tree(self, X, y, depth):
        
        n_sample, n_feature = np.shape(X)

        if self.min_sample_spilt is not None:
            if depth == self.max_depth or n_sample < self.min_sample_spilt:
                most_label = np.argmax(np.bincount(y))            
                return TreeNode(value = most_label)
            
        else:
            if depth == self.max_depth:
                most_label = np.argmax(np.bincount(y))            
                return TreeNode(value = most_label)
            

        best_feature_index, best_threshold = self.get_best_feature(X, y)

        if best_feature_index is None:
            most_label = np.argmax(np.bincount(y))            
            return TreeNode(value = most_label)
        
        #print(np.sh(X[best_index_set[0]]))

        right_indices = X[:, best_feature_index] <= best_threshold
        left_indices = ~right_indices
        
        right = self._build_tree(X[right_indices], y[right_indices], depth+1)
        left = self._build_tree(X[left_indices], y[left_indices], depth+1)
        
        return TreeNode(feature_index=best_feature_index, threshold=best_threshold, right=right, left=left)
    

    def _gini(self, y):

        if len(y) == 0:
            return 0.0
        
        p = np.bincount(y) / len(y)

        return 1 - np.sum(p **2)
    

    
    def get_best_feature(self, X, y):
        
        best_gain = 0
        best_feature_index = None
        best_threshold = None

        n_feature = X.shape[1]
        
        #整體
        currentscore = self._gini(y)


        for col_index in range(n_feature):

            feature_values = np.unique(X[:,col_index])

            for threshold in feature_values:

                right_indices = X[:, col_index] <= threshold
                left_indices = ~right_indices
                
                if self.min_sample_spilt is not None:
                    if len(right_indices) < self.min_sample_spilt or len(left_indices) < self.min_sample_spilt:
                        continue

                p = float(len(right_indices)) / len(y)

                gain = currentscore - p * self._gini(y[right_indices]) - (1-p) * self._gini(y[left_indices])

                if gain > best_gain:

                    best_feature_index = col_index
                    best_threshold = threshold

        return best_feature_index, best_threshold


    def predict(self, X):

        predictions = [self._predict(x, self.tree) for x in X]

        return predictions
    
    def _predict(self, x, node):

        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.right)
        else:
            return self._predict(x, node.left)
        

    def score(self, xtest, ytest):

        predictions = self.predict(xtest)

        acc = np.sum(predictions == ytest) / len(ytest)

        print(f"The Accuracy of Test Data : {acc : .2%}")

    
 