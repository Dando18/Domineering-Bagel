'''
Daniel Nichols
May 2021
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utilities import *
from ml_utilities import LSTM, Dense
from dataset import get_midi_dataset, make_sequences, make_dense_data
from argparse import ArgumentParser


def read_midi_files_and_build_dataset(data_root):
    """ build a dataset based on the midi files in data_root
        Args:
            data_root: directory with .mid files
    """
    import glob
    import os

    DEFAULT_TRACK_IDX = 1
    tracks = []

    for fpath in glob.glob(os.path.join(data_root, '*.mid')):
        midi_data = read_midi_file(fpath)
        tracks.append(midi_data.tracks[DEFAULT_TRACK_IDX])
    
    return get_midi_dataset(tracks)


def get_model(type='lstm', backend='torch', **kwargs):
    """ Create the model.
    """
    assert type in ['lstm', 'dense']
    assert backend in ['torch', 'tf']

    if type == 'lstm' and backend == 'torch':
        model = _get_torch_lstm_model(**kwargs)
    elif type == 'dense' and backend == 'torch':
        model = _get_torch_dense_model(**kwargs) 
    
    return model


def _get_torch_lstm_model(input_dim=5, hidden_units=512, num_layers=1, dropout=0):
    model = LSTM(input_dim=input_dim, hidden_units=hidden_units, num_layers=num_layers, batch_first=True, dropout=dropout)
    return model


def _get_torch_dense_model(input_dim=100, hidden_units=512, num_layers=1, dropout=0):
    model = Dense(input_dim=input_dim, hidden_units=hidden_units, num_layers=num_layers, dropout=dropout)
    return model


def _train_torch_lstm_model(model, dataset, epochs=5, split_loss_alpha=0.5):

    def split_loss(predicted, actual):
        # alpha*c + (1-alpha)*r
        # where c is the crossentropy of first two values
        # and r is MSE or norm of last 3
        # TODO -- make the values 2 and 3 dynamic
        alpha = split_loss_alpha
        categorical_loss = nn.CrossEntropyLoss()(predicted[0:2].view(1,-1), actual[0:2].argmax().view(1))
        print(predicted[:2])
        print(actual[:2])
        regression_loss = torch.nn.MSELoss()(predicted[2:], actual[2:])
        return alpha*categorical_loss + (1.0-alpha)*regression_loss

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        running_loss = 0.0
        type_acc = 0
        steps = 0
        for seq, target in zip(*make_sequences(dataset[0], seq_len=20)):
            seq, target = torch.tensor(seq).float(), torch.tensor(target).float()

            model.zero_grad()

            pred = model.forward(seq)

            #print('{}\t{}'.format(str(pred), str(target)))
            if pred[:2].argmax() == target[:2].argmax():
                type_acc += 1

            loss = split_loss(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        print('epoch {}: loss = {}  type accuracy = {}%'.format(epoch, running_loss/steps, type_acc/steps*100.0))



def _train_torch_dense_model(model, dataset, epochs=100, split_loss_alpha=0.8, seq_length=10, batch_size=16):

    all_sequences = []
    all_targets = []
    for ds in dataset:
        sequences, targets = make_dense_data(ds, seq_len=seq_length)
        all_sequences.append(torch.tensor(sequences).float())
        all_targets.append(torch.tensor(targets).float())
    
    sequences = torch.cat(all_sequences)
    targets = torch.cat(all_targets)


    def split_loss(predicted, actual):
        # alpha*c + (1-alpha)*r
        # where c is the crossentropy of first two values
        # and r is MSE or norm of last 3
        # TODO -- make the values 2 and 3 dynamic
        alpha = split_loss_alpha
        categorical_loss = nn.CrossEntropyLoss()(predicted[:,0:2], actual[:,0:2].argmax(dim=1))
        regression_loss = torch.nn.MSELoss()(predicted[:, 2:], actual[:, 2:])
        return alpha*categorical_loss + (1.0-alpha)*regression_loss

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    train_dataset = TensorDataset(sequences, targets)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        type_acc = 0
        regress_error = 0.0
        steps = 0
        for batch, target in train_dataloader:
            model.zero_grad()

            pred = model.forward(batch)

            type_acc += torch.sum((pred[:,0:2].argmax(dim=1) == target[:,0:2].argmax(dim=1)).int())
            regress_error += torch.sum( (pred[:,2:] - target[:,2:]) ** 2 )

            loss = split_loss(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1
        
        total_loss = running_loss / float(steps)
        total_type_acc = type_acc / float(steps*batch_size)
        total_regress_error = regress_error / float(steps*batch_size)
        print('Epoch {}: loss = {:.6}  type_acc = {:.4}%  regress_error = {:.6}'.format(epoch, total_loss, total_type_acc*100.0, total_regress_error))



def train(model, dataset, type='lstm', backend='torch', **kwargs):
    """ Train the model
    """
    assert type in ['lstm', 'dense']
    assert backend in ['torch', 'tf']

    if type == 'lstm' and backend == 'torch':
        _train_torch_lstm_model(model, dataset, **kwargs)
    elif type == 'dense' and backend == 'torch':
        _train_torch_dense_model(model, dataset, **kwargs)


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-root', type=str, help='root of midi files to use as input data')
    parser.add_argument('--save-dataset', type=str, help='save the created dataset')
    parser.add_argument('--use-cached-dataset', type=str, help='use an existing dataset')
    parser.add_argument('-s', '--seed', type=int, default=1, help='torch seed')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'dense'], help='train with LSTM or dense network')
    parser.add_argument('--hidden-units', type=int, default=256, help='hiddens units in model')
    parser.add_argument('--num-layers', type=int, default=1, help='layers in model')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout in model')
    parser.add_argument('--seq-length', type=int, default=20, help='lstm sequence length')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--generate', action='store_true', help='whether to generate a new midi')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # PREPARE DATA SECTION
    print('getting dataset...')
    if args.use_cached_dataset:
        print('not supported yet')
    else:
        dataset = read_midi_files_and_build_dataset(args.data_root)


    # PREPARE MODEL SECTION
    print('creating model...')
    model = get_model(type=args.model_type, hidden_units=args.hidden_units, num_layers=args.num_layers, dropout=args.dropout)

    # TRAIN SECTION
    train(model, dataset, type=args.model_type, epochs=args.epochs, seq_length=args.seq_length)

    # GENERATE
    


if __name__ == '__main__':
    main()
