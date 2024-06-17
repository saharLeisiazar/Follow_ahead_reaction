import torch
import torch.nn as nn

class LSTMModel2D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel2D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.last = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.last(out)
        return out
    

class prob_dist():
    
    def __init__(self, model_dir):
       
        self.model = torch.load(model_dir).cuda()
        self.model.eval()

    def forward(self, history):
        history = torch.tensor(history).float().cuda()
        history -= torch.clone(history[-1])
        history = history.unsqueeze(0)
        out = self.model(history).detach().squeeze().cpu().numpy()

        return {'left': out[0], 'straight': out[1], 'right': out[2]}
       