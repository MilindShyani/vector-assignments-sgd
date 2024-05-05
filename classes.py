from utils import *


class FCC(nn.Module):
    def __init__(self, vocab_size, hidden_dim : list, output_dim,device,optimal_loss = 0) -> None:
        super().__init__()
        dims = hidden_dim + [output_dim]
        self.V = vocab_size        
        self.layers = nn.ModuleList()
        self.reslayers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.device = device
        self.optimal_loss = optimal_loss
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(self.V,dims[0])
        
        for i in range(len(dims)-2):
            self.layers.append(nn.Linear(dims[i],dims[i+1]))
            self.reslayers.append(nn.Linear(dims[i],dims[i+1]))
        self.layers.append(nn.Linear(dims[-2],dims[-1]))
    
    def forward(self,x):
        x = self.embed(x)        
        
        for layer, reslayer  in zip(self.layers[:-1], self.reslayers):
            y = layer(x)
            x = self.relu(y) + reslayer(x)                    
        x = self.layers[-1](x) 
        x = x / torch.norm(x,dim=-1,keepdim=True)
        
        return x
    

    def train(self,x,epochs,lr,batch_size):
        optimizer = torch.optim.Adam(list(self.layers.parameters()) + list(self.reslayers.parameters()) + list(self.embed.parameters()), lr=lr)
        for e in range(epochs):
            perm = torch.randperm(len(x))
            x = x[perm]            
            labels = (einops.rearrange(x, " i -> 1 i") == einops.rearrange(x, " i -> i 1")).float()
            labels = 2*labels - 1
            
            for b in range(0,len(x),batch_size):
                output = self.forward(x[b:b+batch_size].to(self.device))
                targets = labels[b:b+batch_size,b:b+batch_size].to(self.device)                                
                out = output @ output.T
                optimizer.zero_grad()  
                loss = torch.mean((targets - out)**2)
                loss.backward()
                optimizer.step()

            print(loss.item())
            if loss.item() <= self.optimal_loss*1.01:
                print(f'Almost hit optimal loss. Early stopping.')
                return 
                
            with torch.inference_mode():
                if e % 2 == 0:
                    out_vec = self.forward(torch.arange(self.vocab_size,device=self.device)).cpu().numpy()                                        
                    theta = np.linspace(0, 2*np.pi, 100);  cx = np.cos(theta); cy = np.sin(theta); fig, ax = plt.subplots(); ax.plot(cx, cy); ax.set_aspect('equal')
                    plt.scatter(np.hstack((np.array(0),out_vec[:,0])), np.hstack((np.array(0),out_vec[:,1])) )
                    plt.savefig("out_vecs")                
                    plt.show()
                    
def make_data(samples, concepts):        
    inp = einops.repeat(torch.arange(0,concepts), " i -> (i b)", b = samples//concepts)
    inp = inp[torch.randperm(len(inp))]
    return inp

def make_zipf_data(most_frequent_size, concepts):        
    tensor_list = []
    for i in range(1,concepts+1):
        samples_i = most_frequent_size // i 
        inp = einops.repeat(torch.tensor([i-1]), " i -> (i b)", b = samples_i)
        tensor_list.append(inp)
    inp = torch.cat(tensor_list)
    inp = inp[torch.randperm(len(inp))]
    return inp

def make_zipf_data_with_exp(most_frequent_size, concepts,exp = 0.5):        
    tensor_list = []
    for i in range(1,concepts+1):
        samples_i = int(most_frequent_size / (i**exp))
        inp = einops.repeat(torch.tensor([i-1]), " i -> (i j)", j = int(samples_i))
        tensor_list.append(inp)
    inp = torch.cat(tensor_list)
    inp = inp[torch.randperm(len(inp))]
    return inp