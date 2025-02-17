# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

__version__ = '0.14.0'
short_version = __version__


def parse_version_info(version_str: str) -> Tuple:
    """Parse version from a string.

    Args:
        version_str (str): A string represents a version info.

    Returns:
        tuple: A sequence of integer and string represents version.
    """
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)

__all__ = ['__version__', 'version_info', 'parse_version_info']
