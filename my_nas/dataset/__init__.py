#pylint: disable=unused-import

from my_nas.utils import getLogger
_LOGGER = getLogger("dataset")

from my_nas.dataset.base import BaseDataset
from my_nas.dataset import cifar10
from my_nas.dataset import ptb
from my_nas.dataset import imagenet
from my_nas.dataset import tiny_imagenet
from my_nas.dataset import cifar100
from my_nas.dataset import svhn
from my_nas.dataset import miniimagenet
from my_nas.dataset import imagenet_downsample

# try:
#     from my_nas.dataset import voc
#     from my_nas.dataset import coco
# except ImportError as e:
#     _LOGGER.warn(
#         ("Cannot import module detection: {}\n"
#          "Should install EXTRAS_REQUIRE `det`").format(e))


AVAIL_DATA_TYPES = ["image", "sequence"]
