"""Test suite for TIMIT."""
import os
from audlib.data.timit import TIMIT


def test_timit():
    SI = TIMIT('/home/xyy/data/timit',
               filt=lambda p: 'SI' in os.path.basename(p).upper(),
               readmode='utterance')
    SX = TIMIT('/home/xyy/data/timit',
               filt=lambda p: 'SX' in os.path.basename(p).upper(),
               readmode='rand-phone')

    print(SX, SI)


if __name__ == "__main__":
    test_timit()
