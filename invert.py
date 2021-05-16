'''
Daniel Nichols
May 2021
'''

import mido
import numpy as np
from argparse import ArgumentParser
from utilities import *


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


    # transform every message
    for idx, msg in enumerate(track):
        if msg.is_meta or not hasattr(msg, 'note'):
            continue
        else:
            diff = (msg.note - center)
            msg.note = center - diff

            # handle notes too big
            if msg.note > 127 or msg.note < 0:
                if out_of_bounds_handler == 'clamp':
                    msg.note = 127 if msg.note > 127 else 0
                elif out_of_bounds_handler == 'drop':
                    msg.velocity = 0

def swap_velocity_and_note(track):
    """ Swap the velocity and note value of each message
    """
    for msg in track:
        if hasattr(msg, 'note') and hasattr(msg, 'velocity'):
            tmp = msg.velocity
            msg.velocity = msg.note
            msg.note = tmp


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
