'''
Daniel Nichols
May 2021
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_units=512, num_layers=1, batch_first=True, dropout=0, output_size=5):
        super().__init__()

        self.hidden_units = hidden_units

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_units, num_layers=num_layers, batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_units, output_size)


    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(1, len(input_seq), -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        #print(predictions)
        return predictions


class Dense(nn.Module):
    def __init__(self, input_dim, hidden_units=1024, num_layers=5, dropout=0.2, output_size=5):
        super().__init__()

        self.num_layers = num_layers
        self.layers = []

        units_count = hidden_units
        in_size = input_dim
        for _ in range(self.num_layers-1):
            self.layers.append(
                nn.Linear(in_size, units_count)
            )
            in_size = units_count
            units_count //= 2

        self.dropout_end = nn.Dropout(dropout)
        self.fc_last = nn.Linear(in_size, output_size)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            x = F.relu(x)
        
        x = self.dropout_end(x)
        x = self.fc_last(x)
        x = torch.cat([F.log_softmax(x[:,0:2], -1), x[:,2:]], dim=1)

        return x
        

