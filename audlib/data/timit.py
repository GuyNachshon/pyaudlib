"""Datasets derived from the TIMIT dataset for phoneme recognition."""
import os
from random import randint

from ..io.audio import audioinfo
from .dataset import AudioDataset, audioread
from .datatype import Audio

# Phoneme table defined in TIMIT's doc
PHONETABLE = {p: i for i, p in enumerate(
    """aa ae ah ao aw ax ax-h axr ay b bcl ch d dcl dh dx eh el em en
        eng epi er ey f g gcl h# hh hv ih ix iy jh k kcl l m n ng nx ow
        oy p pau pcl q r s sh t tcl th uh uw ux v w y z zh""".split()
    )}
VOWELS = """iy ih eh ey ae aa aw ay ah ao oy ow uh uw ux er ax ix axr 
            ax-h""".split()
SEMIVOWELS = "l r w y hh hv el".split()


def phnread(path):
    """Read a .PHN (or .WRD) file.

    Format:
        0 3050 h#
        3050 4559 sh
        4559 5723 ix
    """
    try:
        seq = []
        with open(path) as fp:
            for line in fp:
                ns, ne, ph = line.rstrip().split()
                seq.append(((int(ns), int(ne)), ph))
        return seq
    except FileNotFoundError:
        print(f"[{path}] does not exist!")
        return None


def txtread(path):
    """Read a .TXT transcript file."""
    try:
        with open(path) as fp:
            for line in fp:
                line = line.rstrip().split()
                ns, ne, tr = line[0], line[1], ' '.join(line[2:])
                transcrpt = (int(ns), int(ne)), tr
        return transcrpt
    except FileNotFoundError:
        print(f"{path} does not exist!")
        return None


def spkrinfo(path, istrain):
    """Load speaker table from file."""
    with open(path) as fp:
        spkrt = {}  # spkrt['spkr'] = int(label)
        ii = 0
        for line in fp:
            if line[0] != ';':  # ignore header
                line = line.rstrip().split()
                sid, train = line[0], line[3].upper() == 'TRN'
                if not (istrain ^ train):
                    spkrt[sid] = ii
                    ii += 1
    return spkrt


def isvowel(phone, semivowels=True):
    """Check if phone is a vowel (or a semivowel)."""
    if semivowels:
        return (phone in VOWELS) or (phone in SEMIVOWELS)
    else:
        return phone in VOWELS


class TIMITSpeech(Audio):
    """A data structure for TIMIT audio."""
    __slots__ = 'speaker', 'gender', 'transcript', 'phonemeseq', 'wordseq',\
                'phone'

    def __init__(self, signal=None, samplerate=None, speaker=None, gender=None,
                 transcript=None, phonemeseq=None, wordseq=None, phone=None):
        super(TIMITSpeech, self).__init__(signal, samplerate)
        self.speaker = speaker
        self.transcript = transcript
        self.phonemeseq = phonemeseq
        self.gender = gender
        self.wordseq = wordseq
        self.phone = phone  # phone label according to _phon_table


class TIMIT(AudioDataset):
    """Generic TIMIT dataset for phoneme recognition.

    The dataset should follow the directory structure below:
    root
    ├── CONVERT
    ├── SPHERE
    └── TIMIT
        ├── DOC
        ├── TEST
        │   ├── DR1
        │   ├── DR2
        │   ├── DR3
        │   ├── DR4
        │   ├── DR5
        │   ├── DR6
        │   ├── DR7
        │   └── DR8
        └── TRAIN
            ├── DR1
            ├── DR2
            ├── DR3
            ├── DR4
            ├── DR5
            ├── DR6
            ├── DR7
            └── DR8
    """
    @staticmethod
    def isaudio(path):
        return path.endswith('.WAV')

    def __init__(self, root, train=True, sr=None, filt=None,
                 readmode='utterance', transform=None, nosilence=False):
        """Instantiate an ASVspoof dataset.

        Parameters
        ----------
        root: str
            The root directory of TIMIT.
        train: bool, optional
            Retrieve training partition?
        sr: int, optional
            Sampling rate in Hz. TIMIT is recorded at 16kHz.
        filt: callable, optional
            Filters to be applied on each audio path. Default to None.
        phone: bool, False
            Read the audio of an phoneme instead of a sentence?
            If True, randomly read ONE phone segment from an audio.
            If False, entire audio will be read.
        transform: callable(TIMITSpeech) -> TIMITSpeech
            User-defined transformation function.

        Returns
        -------
        An class instance `TIMIT` that has the following properties:
            - len(TIMIT) == number of usable audio samples
            - TIMIT[idx] == a TIMITSpeech instance

        See Also
        --------
        TIMITSpeech, dataset.AudioDataset, datatype.Audio

        """
        self.train = train
        # hard-coded phone labels
        self._phon_table = PHONETABLE
        self._spkr_table = spkrinfo(
            os.path.join(root, 'TIMIT/DOC/SPKRINFO.TXT'), train)
        if train:
            audroot = os.path.join(root, 'TIMIT/TRAIN')
        else:
            audroot = os.path.join(root, 'TIMIT/TEST')
        self.sr = sr
        self.nosilence = nosilence

        def _read(path):
            """Different options to read an audio file."""
            pbase = os.path.splitext(path)[0]
            gsid = pbase.split('/')[-2]
            gender, sid = gsid[0], gsid[1:]
            assert sid in self._spkr_table
            sid = self._spkr_table[sid]
            phoneseq = phnread(pbase+'.PHN')
            if readmode == 'utterance':  # read entire utterance
                wrdseq = phnread(pbase+'.WRD')
                transcrpt = txtread(pbase+'.TXT')
                if self.nosilence:
                    ns, ne = wrdseq[0][0][0], wrdseq[-1][0][1]
                    sig, sr = audioread(path, sr=self.sr, start=ns, stop=ne)
                    pseq = []
                    for (ss, ee), pp in phoneseq:
                        ss -= ns
                        ee -= ns
                        if ss >= 0:  # The "silence" part has labels.
                            pseq.append(((ss, ee), pp))
                    phoneseq = pseq
                else:
                    sig, sr = audioread(path, sr=self.sr)

                return TIMITSpeech(sig, sr, speaker=sid, gender=gender,
                                   transcript=transcrpt, phonemeseq=phoneseq,
                                   wordseq=wrdseq,
                                   phone=[(t, self._phon_table[p])
                                          for t, p in phoneseq])
            elif readmode == 'rand-phoneme':
                # Ignore nosilence here
                (ns, ne), pp = phoneseq[randint(0, len(phoneseq)-1)]
                sig, sr = audioread(path, sr=self.sr, norm=True,
                                    start=ns, stop=ne)
                return TIMITSpeech(sig, sr, speaker=sid, gender=gender,
                                   phone=self._phon_table[pp])
            else:
                raise NotImplementedError

        super(TIMIT, self).__init__(
            audroot, filt=self.isaudio if not filt else lambda p:
                self.isaudio(p) and filt(p), read=_read, transform=transform)

    def __repr__(self):
        """Representation of TIMIT."""
        return r"""{}({}, sr={}, transform={})
        """.format(self.__class__.__name__, self.root, self.sr, self.transform)

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        spkr_appeared = set([])
        for path in self._filepaths:
            sid = path.split('/')[-2][1:]
            assert sid in self._spkr_table, f"{sid} not a valid speaker!"
            spkr_appeared.add(sid)
        phoncts = {p: 0 for p in self._phon_table}
        mindur = {p: 100 for p in self._phon_table}
        maxdur = {p: 0 for p in self._phon_table}
        for path in self._filepaths:
            sr = audioinfo(os.path.join(self.root, path)).samplerate
            path = os.path.join(self.root, os.path.splitext(path)[0]+'.PHN')
            for (ts, te), pp in phnread(path):
                assert pp in phoncts, f"[{pp}] not in phone dict!"
                phoncts[pp] += 1
                dur = (te - ts) / sr * 1000
                if mindur[pp] > dur:
                    mindur[pp] = dur
                if maxdur[pp] < dur:
                    maxdur[pp] = dur
        totcts = sum(v for p, v in phoncts.items())
        report = """
            +++++ Summary for [{}] partition [{}] +++++
            Total [{}] valid files to be processed.
            Total [{}/{}] speakers appear in this set.
            [Phoneme]: [counts], [percentage], [min-max duration (ms)]\n{}
        """.format(self.__class__.__name__, 'train' if self.train else 'test',
                   len(self._filepaths), len(spkr_appeared),
                   len(self._spkr_table),
                   "\n".join(
                       f"\t\t[{p:>4}]: [{c:>4}], [{c*100/totcts:2.2f}%], [{mindur[p]:.1f}-{maxdur[p]:.0f}]"
                       for p, c in phoncts.items()))

        return report


# Some filter functions

def utt_no_shorter_than(path, duration, unit='second'):
    """Check for utterance duration after silence removal."""
    pbase = os.path.splitext(path)[0]
    wrdseq = phnread(pbase+'.WRD')
    dur = wrdseq[-1][0][1] - wrdseq[0][0][0]
    if unit == 'second':
        sr = audioinfo(path).samplerate
        return dur / sr >= duration
    elif unit == 'sample':
        return dur >= duration
    else:
        raise ValueError(f"Unsupported unit: {unit}.")
