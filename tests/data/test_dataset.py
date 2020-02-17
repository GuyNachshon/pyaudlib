import os

from audlib.data.dataset import LongFile

HOME = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"../../../")


def test_LongFile():
    data = LongFile(os.path.join(HOME, 'samples/welcome16k.wav'), 1, 1)
    assert data[0].signal.shape[0] == data[1].signal.shape[0]


if __name__ == '__main__':
    test_LongFile()
