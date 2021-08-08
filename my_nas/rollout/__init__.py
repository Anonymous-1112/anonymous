"""
Rollouts, the inferface of different components in the NAS system.
"""
# pylint: disable=unused-import

from my_nas.utils.exception import expect
from my_nas.rollout.base import (
    BaseRollout,
    Rollout,
    DifferentiableRollout
)
from my_nas.rollout.mutation import (
    MutationRollout,
    CellMutation,
    Population,
    ModelRecord
)
from my_nas.rollout.dense import (
    DenseMutationRollout,
    DenseMutation,
    DenseDiscreteRollout
)
from my_nas.rollout.ofa import (
    MNasNetOFASearchSpace,
    MNasNetOFARollout,
    SSDOFASearchSpace,
    SSDOFARollout
)

from my_nas.rollout.compare import (
    CompareRollout
)

from my_nas.rollout.general import (
    GeneralSearchSpace,
    GeneralRollout
)

from my_nas.rollout.wrapper import (
    WrapperSearchSpace,
    WrapperRollout,
    GermWrapperSearchSpace
)

def assert_rollout_type(type_name):
    expect(type_name in BaseRollout.all_classes_(),
           "rollout type {} not registered yet".format(type_name))
    return type_name
