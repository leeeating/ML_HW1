import numpy as np

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
        y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]

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



class KNNClassifier(object):

    def __init__(self, K=3, method = None):
        self.k = K
        assert method in ["euclidean ", "manhattan", "cosine"], "Method  Not in ['euclidean' , 'manhattan', 'cosine']"

    
    def fit(self, X, y ):
        self.x = X
        self.y = y


def _euclidean(self,X)

    


