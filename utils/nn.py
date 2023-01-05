from torch import linalg
from torch import nn



class Unflatten(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(-1, *self.shape)
        
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * 128, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 3 * 3 * 128),
            Unflatten((128, 3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 5, 2, 2, 1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        gen = self.decoder(x)
        return x, gen
    
    def num_param(self):
        return sum(p.numel() for p in self.parameters()) / 1e6    
    

class DEKM(nn.Module):
    def __init__(self, encoder, center=None):
        super().__init__()
        self.encoder = encoder
        self.center = center

    def forward(self, x):
        x = self.encode(x)
        eu_dist = self.get_distance(x)
        return eu_dist
    
    def encode(self, x):
        return self.encoder(x)
    
    def get_distance(self, x):
        return linalg.norm(x[:, None, :] - self.center, dim=2)
    
    def num_param(self):
        return sum(p.numel() for p in self.parameters()) / 1e6    

