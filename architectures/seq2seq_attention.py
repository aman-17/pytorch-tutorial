import torch 
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional =True, dropout = p)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden, cell) = self.rnn(embedding)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim = 2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim = 2))
        return encoder_states, hidden, cell 


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout = p)
        self.energy = nn.Linear(hidden_size*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.fc_hidden = nn.Linear(hidden_size, output_size)

    
    def forward(self, x, encoder_states, hidden, cell):

        x =x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x)) # (1, N, embedding_size)
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)
        # (seq_length, N, 1)
        attention = attention.permute(1,0,2)
        # (N, 1, seq_length)
        encoder_states = encoder_states.permute(1,0,2)
        # (N, seq_length, hidden_size*2)
        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2) # (1, N, hidden_size*2)
        rnn_input = torch.cat((context_vector, embedding))
        outputs, (hidden, cell) = self.rnn(embedding)
        predictions =self.fc(outputs)
        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, t_r = 0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        encoder_states, hidden, cell = self.encoder(source)
        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best = output.argmax(1)
            x = target[t]  if random.random() < t_r else best
        return outputs

