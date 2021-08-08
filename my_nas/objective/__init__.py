"""
Objectives
"""
#pylint: disable=unused-import

from my_nas.utils import getLogger as _getLogger

from my_nas.objective.base import BaseObjective
from my_nas.objective.image import ClassificationObjective, CrossEntropyLabelSmooth
from my_nas.objective.flops import FlopsObjective
from my_nas.objective.language import LanguageObjective
from my_nas.objective.ofa import OFAClassificationObjective
from my_nas.objective.zeroshot import ZeroShot

