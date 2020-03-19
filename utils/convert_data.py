import numpy as np
import logging

logger = logging.getLogger(__name__)

def convert(in_data, from_unit=None, to_unit=None):
    """
    Converts the in_data from unit 'from_unit' to 'to_unit'.

    :param in_data:
        np.array
    :param from_unit:
        str
    :param to_unit:
        str
    :return:
        np.array
    """
    # Test if units are specified
    if from_unit is None:
        logger.warning("No 'from_unit' specified, conversion failed")
        return
    elif to_unit is None:
        logger.warning("No 'to_unit' specified, conversion failed")
        return

    # Start converting
    if (from_unit == 'us/ft') and (to_unit == 'm/s'):
        return 1. / (3.28E-6 * in_data)

    elif (from_unit == 'us/ft') and (to_unit == 'km/s'):
        return 1. / (3.28E-3 * in_data)

    else:
        wrn_txt = "No valid combination of units specified (from {} to {}), conversion failed".format(
            from_unit, to_unit)
        logger.warning(wrn_txt)
        raise Warning(wrn_txt)

