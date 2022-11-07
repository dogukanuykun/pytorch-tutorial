import torch
import torch.nn as nn
import numpy as np
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # 4x1
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

#model = nn.Linear(input_size, output_size)

class LinarRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinarRegression, self).__init__()

        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self,x):
        return self.lin(x)

model = LinarRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

"""
w = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x 
"""   
"""
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()
"""

learning_rate = 0.1
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


for epoch in range(n_iters):
    y_pred = model(X) 

    l = loss(Y, y_pred)

    #dw = gradient(X, Y, y_pred)
    l.backward() # dl/dw

    optimizer.step()

    #w -= learning_rate * dw
    """    
    with torch.no_grad():
    w -= learning_rate * w.grad
    """

    #w.grad.zero_()
    optimizer.zero_grad()

    if(epoch % 10 == 0):
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(model(X_test))


