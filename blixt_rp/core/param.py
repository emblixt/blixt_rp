# Simple class for holding parameters in blixt_rp

# from dataclasses import dataclass
import numpy as np


# @dataclass
class Param(object):
    """
    Data class to hold the different parameters used in blixt_rp
    """
    def __init__(self,
                 value,
                 name=None,
                 unit=None,
                 desc=None):
        # if name is not None:
        #     self.name = name.lower()
        # else:
        self.name = name
        self.value = value
        self.unit = unit
        self.desc = desc
    # name: str
    # value: float
    # unit: str
    # desc: str

    def __repr__(self):
        # if len(self.desc) == 0:
        #     return '{} = {} [{}]'.format(self.name, self.value, self.unit)
        # else:
        #     return '{} = {} [{}], {}'.format(self.name, self.value, self.unit, self.desc)
        return '{}'.format(self.value)

    # def __setattr__(self, key, value):
    #     """
    #     Force name to be in lower case
    #     """
    #     if key == 'name' and value is not None:
    #         super(Param, self).__setattr__(key, value.lower())
    #     else:
    #         super(Param, self).__setattr__(key, value)

    def __len__(self):
        if isinstance(self.value, float):
            return 1
        elif isinstance(self.value, int):
            return 1
        elif isinstance(self.value, np.ndarray):
            return len(self.value)
        else:
            raise TypeError('Object of type {} has no len()'.format(type(self.value)))


def test():
    x1 = Param(name='X1', value=4.5)
    x2 = Param(name='x2', value=np.linspace(2, 4, 10))
    x2.name = 'XXX'
    x3 = Param(name='x3', value=2)
    x4 = Param(np.linspace(1, 10, 100), 'My Name', unit='m')
    print(x1.name, x2.name, x3.name, x4.name)
    print(x1.unit, x2.unit, x3.unit, x4.unit)
    print(len(x1), len(x2), len(x3), len(x4))


if __name__ == '__main__':
    test()

