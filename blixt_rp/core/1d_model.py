class Model:
    """
    Object that holds a subsurface model with N layers
    """

    def __init__(self,
                 model_type=None,
                 depth_to_top=None,
                 layers=None,
                 domain=None):
        """
        :param model_type:
            str
            '1D' is the only option
        :param depth_to_top:
            float
            depth to top in seconds or meters
        :param layers:
            list of layers, each a Layer object
        :param domain:
            string
            'TWT' (time in seconds) or 'Z' (depth in meters)
        """

        if model_type is None:
            self.model_type = '1D'
        else:
            self.model_type = model_type

        if depth_to_top is None:
            self.depth_to_top = 2.0
        else:
            self.depth_to_top = depth_to_top

        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        if domain is None:
            self.domain = 'TWT'
        else:
            self.model_type = model_type

    def __str__(self):
        return '{} model in {} domain with {} layers'.format(self.model_type, self.domain, len(self.layers))

    def __len__(self):
        return len(self.layers)

    def append(self, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        self.layers.append(layer)

    def insert(self, index, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        self.layers.insert(index, layer)


class Layer:

    def __init__(self,
                 vp=3600.):
        self.vp = vp


def test():
    first_layer = Layer(3000)
    second_layer = Layer(4000)

    m = Model(layers=[first_layer, second_layer])
    print(m)
    m.append(first_layer)
    print(m)


if __name__ == '__main__':
    test()