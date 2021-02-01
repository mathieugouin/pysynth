# To make print working for Python2/3
from __future__ import print_function
from __future__ import division

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

F_SAMPLE = 44100.0
NOTE_DURATION = 0.95 # % of total duration to not have legato


# https://en.wikipedia.org/wiki/Octave
NOTES = {
    'C':  -9,

    'C#': -8,
    'Db': -8,

    'D':  -7,

    'D#': -6,
    'Eb': -6,

    'E':  -5,

    'F':  -4,

    'F#': -3,
    'Gb': -3,

    'G':  -2,

    'G#': -1,
    'Ab': -1,

    'A':  0,  # A 440 Hz is A4

    'A#': 1,
    'Bb': 1,

    'B':  2,
}


def note_freq(offset):
    return 440.0 * 2.0 ** (offset / 12.0)


def note_offset(note_name):
    """C4, Bb3, F#5"""
    m = re.match(r'([A-G][b#]?)(\d)', note_name.strip())
    if m:
        note = m.group(1)
        octave = int(m.group(2))
        return (octave - 4) * 12 + NOTES[note]
    return 0  # TBD error


def get_note_freq(note_or_offset):
    if type(note_or_offset) == int:
        return note_freq(note_or_offset)
    if type(note_or_offset) == str:
        return note_freq(note_offset(note_or_offset))


def get_hammond_note(setting, note, t):
    DB = [-12, 7, 0, 12, 12+7, 2*12, 2*12+4, 2*12+7, 3*12]

    # DB setting 0-8
    # 0 = OFF
    # 8 = Full ON
    # 1 step = -3 dB
    # power_db = 20 * log10(amp / amp_ref)
    # power_db = 20 * np.log10(a / 1)
    # a = 10 ** (db / 20)
    def db_amplitude(s):
        return (s != 0) * 10 ** (10 * np.log10(1.0/2) * (8.0 - s) / 20)

    if type(note) == str:
        offset = note_offset(note)
    else:
        offset = note

    sound = t * 0.0
    amplitude = [db_amplitude(int(i)) for i in re.sub('\s', '', setting)]
    if len(amplitude) == len(DB):
        for i in range(len(DB)):
            sound = sound + amplitude[i] * np.sin(2 * np.pi * get_note_freq(offset + DB[i]) * t)
    return sound


# TBD not my code...
def ASD_envelope(nSamps, tAttack, tRelease, susPlateau, kA, kS, kD):

    # 0 to 1 over N samples, weighted with w
    def weighted_exp(N, w):
        t = np.linspace(0, 1, N)
        E = np.exp(w * t) - 1
        E /= max(E)
        return E

    # number of samples for each stage
    sA = int(nSamps * tAttack)
    sD = int(nSamps * tRelease)
    sS = nSamps - sA - sD

    A = weighted_exp(sA, kA)
    S = weighted_exp(sS, kS)
    D = weighted_exp(sD, kD)

    A = A[::-1]
    A = 1.0 - A

    S = S[::-1]
    S *= 1.0 - susPlateau
    S += susPlateau

    D = D[::-1]
    D *= susPlateau

    env = np.concatenate([A,S,D])

    return env


def gen_note(note, duration):
    """duration in seconds"""
    t = np.arange(0, duration, 1 / F_SAMPLE)

    if note == 'R':  # Rest
        return t * 0.0

    #                (nSamps, tAttack, tRelease, susPlateau, kA, kS, kD):
    env = ASD_envelope(len(t), 0.05, 0.2, 0.8, 3, 2, 3)
    if False: # Simple note
        f = get_note_freq(note)
        y = np.sin(2 * np.pi * f * t)
    else:
        y = get_hammond_note('67 8404 231', note, t)
    y *= env

    return y


def gen_chord(notes, duration):
    return sum([gen_note(n, duration) for n in notes])


def get_duration(note_fraction, tempo):
    return note_fraction * 60.0 / tempo


def play(music, tempo):
    song = np.zeros(1)
    for line in music.splitlines():
        line = line.strip()
        if line:
            print(line)
            s = line.split()
            if len(s) >= 2:
                fraction = eval(s[-1])
                duration = get_duration(fraction, tempo)
                song = np.append(
                    song,
                    gen_chord(s[:-1], duration)
                )

    # Normalize
    m = max(song)
    if m > 0:
        song /= m

    return song


def write_wav(filename, sound):
    sound = sound / sound.max() * np.iinfo(np.int32).max
    write(filename, F_SAMPLE, sound.astype(np.int32))


def _main():
    env = ASD_envelope(10000, 0.05, 0.2, 0.8, 3, 2, 5)
    #plt.plot(env)
    #plt.show()

    print(note_freq(-12))
    print(note_freq(0))
    print(note_freq(1))
    print(note_freq(12))

    for n in NOTES:
        print('{}\t{}'.format(n, note_freq(NOTES[n])))

    for n in ["C4", "Bb3", "F#5", "G2", "A4"]:
        print('{}\t{}'.format(n, get_note_freq(n)))

    d = 30 * 1 / F_SAMPLE
    d = 0.1 # 100 ms
    #d = 6 * 60 # 6 minutes

    print(gen_chord([0, 4, 7, 10], d))
    print(gen_chord(['A4', 'C#5', 'E5', 'G5'], d))

    tempo = 120
    for i in [4, 3, 2, 1, 0.5, 0.25, 0.125]:
        print('{}\t{}'.format(i, get_duration(i, tempo)))

    print(gen_note('A4', get_duration(.125, tempo)))

    # In beat units with respect to the tempo
    # R is rest
    d = 1/8
    music = """

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    
    C4 4
    R 4

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)

    C4 2
    D4 2
    R 4

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)

    C4 1
    D4 1
    E4 1
    F4 1
    R 4

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)

    C4 1/2
    D4 1/2
    E4 1/2
    F4 1/2
    G4 1/2
    A4 1/2
    B4 1/2
    C5 1/2
    R 4

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)

    C4 1/8
    D4 1/8
    E4 1/8
    F4 1/8
    G4 1/8
    A4 1/8
    B4 1/8
    C5 1/8
    D5 1/8
    E5 1/8
    F5 1/8
    G5 1/8
    A5 1/8
    B5 1/8
    C6 1/8
    D6 1/8
    R 4

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)

    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    B4 1/16
    R 4

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)

    C3 E3 G3 4

    R 4

    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)
    F#5 1/8
    R (1-1/8)

    B4 .5
    B4 1/2
    B4 G4 1
    B4 .5
    B4 .5
    B4 G4 1

    B4 .5
    D5 .5
    G4 1/2+1/4
    A4 1/8
    B4 G4 D4 2

    C5 .5
    C5 .5
    C5 1/2+1/4
    D5 1/8
    C5 .5
    B4 .5
    B4 .5
    B4 .5
    B4 .5
    A4 .5
    A4 .5
    B4 .5
    A4 1
    D5 1

    R 1
    """

    #print(music)
    sound = play(music, 120)
    #plt.plot(sound)
    #plt.show()
    #print(sound)
    write_wav("sound.wav", sound)


if __name__ == '__main__':
    _main()
