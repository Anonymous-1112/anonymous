"""
Evaluators: evaluator components is responsible for evaluating the performance of a candidate
network.
"""

#pylint: disable=unused-import, unused-wildcard-import, wildcard-import

from my_nas.evaluator.base import BaseEvaluator
from my_nas.evaluator.mepa import MepaEvaluator
from my_nas.evaluator.shared_weight import *
from my_nas.evaluator.tune import TuneEvaluator
from my_nas.evaluator.compare import ArchNetworkEvaluator
from my_nas.evaluator.arch_network import *
from my_nas.evaluator.few_shot_shared_weight import *
