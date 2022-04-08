import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, hidden_size, input_size = 256, output_size = 10, std = 1e-4):
        
        self.params = {}
        
        # self.params should contain self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        # TODO ：initialize the parmas
        #
        self.params['W1'] = np.random.randn(input_size, hidden_size) * std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * std
        self.params['b2'] = np.zeros(output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        return
    
    def forward_pass(self, X, y = None, wd_decay = 0.0):
    
        loss = None
        predict = None
        
        #
        # TODO ： finish hidden layer and class layer
        #
        self.out1 = np.dot(X, self.params['W1']) + self.params['b1']
        self.out2 = np.maximum(self.out1, 0)
        self.out3 = np.dot(self.out2, self.params['W2']) + self.params['b2']
        self.out4 = np.exp(self.out3)
        self.out5 = self.out4 / np.sum(self.out4, axis=1).reshape(X.shape[0], 1)
        out = self.out5

        if y is None:
            #
            # TODO : Return prediction. prediction should have the same shape of y, array [N].
            #      Prediction should be 0-9
            
            predict = np.argmax(out, axis=1)
            return predict

        else:
            #
            # TODO : Return Loss
            #
            # Note that y is not one-hot matrix !!!
            loss = np.mean(-np.log(out[range(len(out)), y])) + wd_decay * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2)) / 2
            return loss
        
    def back_prop(self, X, y, wd_decay = 0.0):
        grads = {}

        #grads should contain grads['W1'] grads['b1'] grads['W2'] grads['b2']
        
        #
        # TODO
        #
        grads = {}
        N = X.shape[0]  # batch size
        delta_k = np.eye(self.output_size)[y]  # convert the ground truth y into one-hot matrix

        grads['W1'] = 1/N * (np.dot(X.T, np.multiply(np.dot((self.out5 - delta_k), self.params['W2'].T), (self.out1 > 0)))) + wd_decay * self.params['W1']
        grads['b1'] = np.mean(np.multiply(np.dot((self.out5 - delta_k), self.params['W2'].T), (self.out1 > 0)), axis=0)
        grads['W2'] = 1/N * (np.dot(self.out2.T, (self.out5 - delta_k))) + wd_decay * self.params['W2']
        grads['b2'] = np.mean(self.out5 - delta_k, axis=0)
        
        return grads
 
    def numerical_gradient(self, X, y, wd_decay = 0.0, delta = 1e-6):
        grads = {}
            
        for param_name in self.params:
            grads[param_name] = np.zeros(self.params[param_name].shape)
            itx = np.nditer(self.params[param_name], flags=['multi_index'], op_flags=['readwrite'])
            while not itx.finished:
                idx = itx.multi_index
                
                #This part will iterate for every params
                #You can use self.parmas[param_name][idx] and grads[param_name][idx] to access or modify params and grads
                
                #
                # TODO
                #
                self.params[param_name][idx] += delta
                f1 = self.forward_pass(X, y, wd_decay)
                self.params[param_name][idx] -= 2 * delta  
                f2 = self.forward_pass(X, y, wd_decay)

                grads[param_name][idx] = (f1 - f2) / 2 / delta

                self.params[param_name][idx] += delta
                
                itx.iternext()
        return grads
    
    def get_acc(self, X, y):
        pred = self.forward_pass(X)
        return np.mean(pred == y)
    
    def train(self, X, y, X_val, y_val,
                learning_rate=0, lr_decay=1,
                momentum=0, do_early_stopping=False, stopping_patience=0,
                wd_decay=0, num_iters=10,
                batch_size=4, verbose=False, print_every=10, do_lr_decay=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        acc_history = []
        val_acc_history = []
        val_loss_history = []
        
        #
        # TODO: training process
        #
        # initialize the velocity for SGD optimizer
        v = {}
        for param_name in self.params:
            v[param_name] = np.zeros(self.params[param_name].shape)

        # lr_scheduler: StepLR
        step_size = 8
        gamma = 0.1
        base_lr = learning_rate

        # early stopping
        val_acc = 0
        best_val_acc = 0
        patience = 7
        es = 0
        

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #
            # TODO: training process
            #
            indexs = np.random.choice(num_train, batch_size)
            X_batch = X[indexs]
            y_batch = y[indexs]

            loss = self.forward_pass(X_batch, y_batch, wd_decay=wd_decay)
            grads = self.back_prop(X_batch, y_batch, wd_decay=wd_decay)

            # StepLR
            if do_lr_decay:
                learning_rate = base_lr * (gamma **((it // iterations_per_epoch) // step_size))

            for param_name in self.params:
                v[param_name] = momentum * v[param_name] - learning_rate * grads[param_name]
                self.params[param_name] += v[param_name]
            
            val_loss = self.forward_pass(X_val, y_val, wd_decay=wd_decay)

            loss_history.append(loss)
            val_loss_history.append(val_loss)
            
            if verbose and it % print_every == 0:
                print('iteration %d / %d: training loss: %f val loss: %f' % (it, num_iters, loss, val_loss))
                print('learning_rate: %.3f' %(learning_rate))
 
            if it % iterations_per_epoch == 0:
                
                train_acc = self.get_acc(X_batch, y_batch)
                val_acc = self.get_acc(X_val, y_val)
                acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                if do_early_stopping:
                    #
                    # TODO: early stopping
                    #
                    # Early stops the training if validation accuracy doesn't improve after a given patience.

                    if val_acc > best_val_acc:
                        es = 0
                        best_val_acc = val_acc
                    else:
                        es += 1

                    if es >= patience:
                        print("Early stopping with best_acc: ", best_val_acc, "and val_acc for this epoch: ", val_acc)
                        break



        return {
          'loss_history': loss_history,
          'val_loss_history': val_loss_history,
          'acc_history': acc_history,
          'val_acc_history': val_acc_history,
        }