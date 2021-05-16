'''
Daniel Nichols
May 2021
'''
import mido
import numpy as np


def vprint(verbose, msg, **kwargs):
    if verbose:
        print(msg, **kwargs)


def read_midi_file(file_path):
    try:
        midi_file = mido.MidiFile(file_path)
    except Exception as e:
        print('Cannot read midi file \'{}\': {}'.format(file_path, str(e)))
        return None
    
    return midi_file


def write_midi_file(midi_data, file_path):
    try:
        midi_data.save(file_path)
    except Exception as e:
        print('Cannot write midi file \'{}\': {}'.format(file_path, str(e)))
    

def print_midi_data(midi_data, print_messages=True, print_meta_messages=False):
    for i, track in enumerate(midi_data.tracks):
        print('Track {}: {}'.format(i, track.name))

        if print_messages:
            for msg in track:
                if msg.is_meta and not print_meta_messages:
                    continue
                print(msg)


def pitch_track(track, n_semitones):
    """ Move every note by 'n_semitones' semitones
    """
    for msg in track:
        if hasattr(msg, 'note'):
            msg.note += n_semitones


def clamp_midi_to_piano_range(track):
    """ Clamps midi values to [21,108]. A0 is 21, C8 is 108.
    """
    for msg in track:
        if not hasattr(msg, 'note'):
            continue

        if msg.note > 108:
            msg.note = 108
        elif msg.note < 21:
            msg.note = 21


def shift_midi_to_piano_range(track):
    """ Tries to move midi to fit in piano range. A0 is 21, C8 is 108
    """
    # calc midi range
    notes_arr = get_notes_array(track)
    min_note, max_note = int(np.min(notes_arr)), int(np.max(notes_arr))

    A0 = 21
    C8 = 21

    # if max_note>108 and min_note>=(A0+12), then we can shift an octave down
    while min_note >= (A0+12) and max_note > 108:
        pitch_track(track, -12)
        min_note -= 12
        max_note -= 12

    # if min_note<21 and max_note<=(C8-12), then we can shift an octave up
    while min_note < 21 and max_note <= (C8-12):
        pitch_track(track, 12)
        min_note += 12
        max_note += 12


def midi_note_to_str(note):
    octave = (note // 12) - 1
    note -= 21
    NAMES = ['A','A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    name = NAMES[note % 12]
    return str(name) + str(octave)


def get_notes_array(track):
    notes = []
    for msg in track:
        if hasattr(msg, 'note'):
            notes.append(msg.note)
    return np.array(notes, dtype=np.int)


def apply_mask(track, semitones, zero_fill=True):
    """ Shift the n-th note in track by n semitones.
    """
    for idx, msg in enumerate(track):
        if hasattr(msg, 'type') and msg.type == 'note_on':

            # find corresponding note_off
            search_idx = idx+1
            matching_msg = None
            while search_idx < len(track):
                new_msg = track[search_idx]

                #vprint((idx == 2347), 'looking at note {}: {}'.format(search_idx, new_msg))

                if (hasattr(new_msg, 'note') and new_msg.note == msg.note) and (new_msg.type == 'note_off' or new_msg.velocity == 0):
                    # found it
                    matching_msg = new_msg
                    break

                search_idx += 1
            
            # modify the note of msg and matching message as necessary
            if matching_msg is not None:
                msg.note += semitones[idx]
                matching_msg.note += semitones[idx]
            else:
                print('Could not find matching note for {} {}'.format(idx, str(msg)))
                pass