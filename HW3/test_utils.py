import numpy as np
import matplotlib.pyplot as plt

from Network import Network

def check_pred(pred):
    
    std_pred = np.load('test_pred.npy')
    
    return np.array_equal(pred, std_pred)

def init_toy_data(num_inputs, input_size, num_classes):
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.random.randint(0, num_classes, num_inputs)
    return X, y

def rel_error(x, y):
    
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def plot_loss_history(loss_history = None, val_loss_history = None):
    handles = []
    if loss_history:
        train, = plt.plot(loss_history, label = 'Train')
        handles.append(train)
    if val_loss_history:
        val, = plt.plot(val_loss_history, label = 'Validation')
        handles.append(val)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(handles=handles)
    plt.show()

    
def plot_acc_history(acc_history = None, val_acc_history = None):
    handles = []
    if acc_history:
        train, = plt.plot(acc_history, label = 'Train')
        handles.append(train)
    if val_acc_history:
        val, = plt.plot(val_acc_history, label = 'Validation')
        handles.append(val)
    plt.title('Acc history')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(handles=handles)
    plt.show()
