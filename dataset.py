'''
Daniel Nichols
May 2021
'''
import mido
import numpy as np
from utilities import *


def get_midi_dataset(tracks, flatten=False, type='numpy', **kwargs):
    """ Take a list of tracks and return a full data set.
        Args:
            flatten: flatten the tracks into a single dataset
            type: One of ['numpy', 'torch', 'tf']. The type of dataset to build.
            one_hot_type: One hot encode the event types.
            include_channel: include channel data.
    """
    assert type in ['numpy']

    dataset = []
    for track in tracks:
        track_dataset = get_midi_dataset_for_track(track, type=type, **kwargs)
        dataset.append(track_dataset)
    
    return np.array(dataset)
    

def get_midi_dataset_for_track(track, type='numpy', one_hot_type=True, include_meta=False, include_channel=False):
    """ Get a dataset of midi data
        Args:
            track: midi track used to build dataset.
            type: One of ['numpy', 'torch', 'tf']. The type of dataset to build.
            one_hot_type: One hot encode the event types.
            include_channel: include channel data.
    """
    data_builder = []

    msg_types = {'note_on': 0, 'note_off': 1}

    for msg in track:
        sample = []
        if msg.is_meta or msg.type not in msg_types.keys():
            # skip these for now
            continue
        else:
            # first add the type info
            if one_hot_type:
                for key in msg_types.keys():
                    sample.append(1 if msg.type == key else 0)
            else:
                sample.append(msg_types[msg.type])
        
            # now add the note, velocity, and time
            sample.append(msg.note)
            sample.append(msg.velocity)
            sample.append(msg.time)

        data_builder.append(sample)

    return np.array(data_builder, dtype=np.float64)


def make_sequences(dataset, seq_len=20):
    """ Make sequences of data
    """
    sequences = []
    targets = []
    
    for i in range(len(dataset)-seq_len-1):
        # make a sequence of [i,i+seq_len) with target i+seq_len
        seq = []
        for j in range(i, i+seq_len):
            seq.append(dataset[j])
        
        sequences.append(seq)
        targets.append(dataset[i+seq_len])
    
    return np.array(sequences), np.array(targets)


def make_dense_data(dataset, seq_len=20):
    """ Similar to make_sequences, but make flat sequences
    """
    sequences, targets = make_sequences(dataset, seq_len=20)

    return np.array(sequences).reshape(len(sequences),-1), targets

