import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    
    '''
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''

    def __init__(self, h, w, n_out):
        
        self.h = h
        self.w = w
        self.n_out = n_out
        
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, n_out)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    def clone(self):
        clone = ConvNet(h=self.h, w=self.w, n_out=self.n_out)
        clone.load_state_dict(self.state_dict())
        return clone
    
class GoalCondConvNet(nn.Module):
    
    def __init__(self, h, w, n_out):
        
        self.h = h
        self.w = w
        self.n_out = n_out
                
        super(GoalCondConvNet, self).__init__()
        self.conv1 = ConvNet(h=self.h, w=self.w, n_out=128)
        self.conv2 = ConvNet(h=self.h, w=self.w, n_out=128)
        
        self.merge = nn.Linear(128*2, n_out)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        return self.merge(torch.cat((x1, x2), axis=-1))
    
    def clone(self):
        clone = GoalCondConvNet(h=self.h, w=self.w, n_out=self.n_out)
        clone.load_state_dict(self.state_dict())
        return clone