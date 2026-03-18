import torch.nn as nn
import torch

class LSTM_Scratch(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear((input_dim + hidden_size), hidden_size * 4)

    def forward(self, x): # x: (B, T, d)
        B, T, d = x.shape
        cell_state = torch.zeros((B, self.hidden_size), device=x.device)
        hidden_state = torch.zeros((B, self.hidden_size), device=x.device)
        
        for t in range(T):
            xt = x[:, t, :]
            combined = torch.cat([xt, hidden_state], dim=1) # (B, (d + hidden_size))
            gates = self.W(combined) # (B, 4 * hidden_size)

            f, i, g, o = gates.chunk(4, dim=1)

            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            cell_state = f * cell_state + i * g
            hidden_state = o * torch.tanh(cell_state)
            
        return hidden_state
    
class Model_Scratch(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout):
        super().__init__()
        self.lstm = LSTM_Scratch(input_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h = self.lstm(x) # (B, hidden_size)
        pred = self.fc(self.dropout(h))
        return pred

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(input_size=config.input_dim, hidden_size=config.lstm_hidden_size, num_layers=config.layers, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.lstm_hidden_size, 1)
    def forward(self, x): # x: (B, seq_length, # of features)
        """
        out - hidden state for all timesteps: (B, seq_length, hidden_size)
        h_n - final hidden state for all layers: (num_layers, B, hidden_size)
        c_n - final cell state for all layers: (num_layers, B, hidden_size)
        h_last - final hidden state for last layer: (B, hidden_size)
        """
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        preds = self.fc(self.dropout(h_last))
        return preds