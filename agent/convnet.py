import torch
import torch.nn as nn
import torch.nn.functional as F

class GridConvNet(nn.Module):
    
    '''
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''

    def __init__(self, h, w, n_out):
        
        self.h = h
        self.w = w
        self.n_out = n_out
        
        super(GridConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(8)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = int(convw) * int(convh) * 8
        self.head = nn.Linear(linear_input_size, n_out)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))
    
    def clone(self):
        clone = GridConvNet(h=self.h, w=self.w, n_out=self.n_out)
        clone.load_state_dict(self.state_dict())
        return clone
    
class ImgConvNet(nn.Module):
    
    '''
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''

    def __init__(self, h, w, n_out):
        
        self.h = h
        self.w = w
        self.n_out = n_out
        
        super(ImgConvNet, self).__init__()
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
        clone = ImgConvNet(h=self.h, w=self.w, n_out=self.n_out)
        clone.load_state_dict(self.state_dict())
        return clone

class GoalCondGridConvNet(nn.Module):
    
    def __init__(self, h, w, n_out):
        
        self.h = h
        self.w = w
        self.n_out = n_out
        
        super(GoalCondGridConvNet, self).__init__()
        self.conv1 = GridConvNet(h=self.h, w=self.w, n_out=64)
        self.conv2 = GridConvNet(h=self.h, w=self.w, n_out=64)
        
        self.h1 = nn.Sequential(*[nn.Linear(64*2, 64), 
                                   nn.ReLU()])
        self.h2 = nn.Sequential(*[nn.Linear(64, 32), 
                                   nn.ReLU()])
        self.h3 = nn.Sequential(*[nn.Linear(32, 16), 
                                   nn.ReLU()])
        self.head = nn.Linear(16, n_out)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = self.h1(torch.cat((x1, x2), axis=-1))
        x = self.h2(x)
        x = self.h3(x)
        return self.head(x)
    
    def clone(self):
        clone = GoalCondGridConvNet(h=self.h, w=self.w, n_out=self.n_out)
        clone.load_state_dict(self.state_dict())
        return clone
    
class GoalCondImgConvNet(nn.Module):
    
    def __init__(self, h, w, n_out):
        
        self.h = h
        self.w = w
        self.n_out = n_out
                
        super(GoalCondImgConvNet, self).__init__()
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
        clone = GoalCondImgConvNet(h=self.h, w=self.w, n_out=self.n_out)
        clone.load_state_dict(self.state_dict())
        return clone