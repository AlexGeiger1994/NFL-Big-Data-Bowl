# pytorch Neural network
class PytorchNetwork(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        
        # first layer
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.CELU()
        
        # second layer
        self.dout2 = nn.Dropout(0.5)
        self.lin2    = nn.Linear(256,512)
        self.relu2 = nn.ReLU()
        
        self.dout3 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        
        self.dout4 = nn.Dropout(0.25)
        self.out = nn.Linear(256, 199)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        
        # input & first layer
        a1  = self.fc1(input_)
        bn1 = self.bn1(a1)
        h1  = self.relu1(bn1)
        
        d2 =self.dout2(h1)
        f2 = self.lin2(d2)
        a2 = self.relu2(f2)
        
        d3 = self.dout3(a2)
        a3 = self.fc3(d3)
        h3 = self.relu3(a3)
        
        d4 = self.dout4(h3)
        a5 = self.out(d4)
        y = self.out_act(a5)
        return a5

