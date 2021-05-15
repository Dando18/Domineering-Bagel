'''
Daniel Nichols
May 2021
'''

import mido
import numpy as np
from numpy.core.overrides import set_module
from argparse import ArgumentParser


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


def invert_track(track, center=63, out_of_bounds_handler='clamp', verbose=False):
    """ Inverts the given track about 'center'.
    Args:
        track: midi file track object from mido
        center: center about which to invert. Can be an integer representing the note. Or can be 
                'middle', 'middle-c', 'first_note', 'middle_note', 'last_note', 'max_note', 'min_note', 'median_note', or 'mean_note'
        out_of_bounds_hander: how to handle notes outside of 0-127. Can be 'clamp' or 'drop'
    """
    assert track is not None
    assert out_of_bounds_handler in ['clamp', 'drop']

    is_center_moving = False
    moving_center_name = ''
    if isinstance(center, str):
        assert center in ['middle', 'middle-c', 'first_note', 'middle_note', 'last_note', 'max_note', 
                            'min_note', 'median_note', 'mean_note', 'pi']
        center_str = center

        notes_arr = get_notes_array(track)
        if center_str == 'middle':
            center = 63
        elif center_str == 'middle-c':
            center = 60
        elif center_str == 'first_note':
            center = notes_arr[0]
        elif center_str == 'middle_note':
            center = notes_arr[len(notes_arr)//2]
        elif center_str == 'last_note':
            center = notes_arr[-1]
        elif center_str == 'max_note':
            center = int(np.max(notes_arr))
        elif center_str == 'min_note':
            center = int(np.min(notes_arr))
        elif center_str == 'median_note':
            center = int(np.median(notes_arr))
        elif center_str == 'mean_note':
            center = int(np.mean(notes_arr))
        elif center_str == 'pi':
            center_str = 'pi'
            center = 3
            is_center_moving = True


    vprint(verbose, 'Inverting around center note {} (midi_val={}).'.format(midi_note_to_str(center), center))

    def _update_center_helper(idx, cur_center, center_type):
        if center_type == 'pi':
            return int(PI_STR[idx])


    # transform every message
    for idx, msg in enumerate(track):
        if msg.is_meta or not hasattr(msg, 'note'):
            continue
        else:
            if is_center_moving:
                center = _update_center_helper(idx, center, center_str)

            diff = (msg.note - center)
            msg.note = center - diff

            # handle notes too big
            if msg.note > 127 or msg.note < 0:
                if out_of_bounds_handler == 'clamp':
                    msg.note = 127 if msg.note > 127 else 0
                elif out_of_bounds_handler == 'drop':
                    msg.velocity = 0


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


def swap_velocity_and_note(track):
    """ Swap the velocity and note value of each message
    """
    for msg in track:
        if hasattr(msg, 'note') and hasattr(msg, 'velocity'):
            tmp = msg.velocity
            msg.velocity = msg.note
            msg.note = tmp


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



def pi_shift(track, sgn='alternate'):
    """ Move the n-th note by the n-th digit of pi semitones.
        Args:
            track: track to modify
            sgn: 'up', 'down', or 'alternate' -- how to move the notes
    """
    from mpmath import mp
    PI_DIGITS = 10000
    mp.dps = PI_DIGITS
    PI_STR = str(mp.pi)

    semitones = np.array(list(map(int, PI_STR.replace('.', ''))))

    if sgn == 'down':
        semitones *= -1
    elif sgn == 'alternate':
        semitones[1::2] *= -1

    apply_mask(track, semitones, zero_fill=True)



def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input midi file')
    parser.add_argument('-o', '--output', type=str, help='output midi file.')
    parser.add_argument('--invert', action='store_true', help='invert the first track')
    parser.add_argument('--invert-center', default=63, help='center to invert around')
    parser.add_argument('--swap-velocity-and-note', action='store_true', help='swap message velocities and note values')
    parser.add_argument('--pi-shift', action='store_true', help='shift notes by PI values')
    parser.add_argument('--clamp-to-piano', action='store_true', help='clamp all notes to piano range')
    parser.add_argument('--shift-to-piano', action='store_true', help='shift all notes to piano range')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
    args = parser.parse_args()

    try:
        args.invert_center = int(args.invert_center)
    except:
        pass

    midi_data = read_midi_file(args.input)

    #print_midi_data(midi_data)

    if args.invert:
        invert_track(midi_data.tracks[1], center=args.invert_center, verbose=args.verbose)

    if args.swap_velocity_and_note:
        swap_velocity_and_note(midi_data.tracks[1])

    if args.pi_shift:
        pi_shift(midi_data.tracks[1], sgn='alternate')

    if args.clamp_to_piano:
        clamp_midi_to_piano_range(midi_data.tracks[1])

    if args.shift_to_piano:
        shift_midi_to_piano_range(midi_data.tracks[1])

    if args.output:
        write_midi_file(midi_data, args.output)


if __name__ == '__main__':
    main()
