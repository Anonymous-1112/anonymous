"""
Built-in tight coupled NAS flows.
"""

from my_nas.utils import getLogger
_LOGGER = getLogger("btc")

try:
    from my_nas.btcs import nasbench_101
except ImportError as e:
    _LOGGER.warn(
        ("Cannot import module nasbench: {}\n"
         "Should install the NASBench 101 package following "
         "https://github.com/google-research/nasbench").format(e))

try:
    from my_nas.btcs import nasbench_201
except ImportError as e:
    _LOGGER.warn(
        ("Cannot import module nasbench_201: {}\n"
         "Should install the NASBench 201 package following "
         "https://github.com/D-X-Y/NAS-Bench-201").format(e))
        
from my_nas.btcs import nasbench_301
