from ._core_trf import Transformer, FeaT, FeedForward, Attention
from ._xr1_cnn import XR1Cnn
from ._mrN_cnn_trf import MR1CnnTrf, MR2CnnTrf
from ._xr1mrN import XR1MR1CnnTrf, XR1MR2CnnTrf
from ._xrNmrMcP import XR1MR2C1CnnTrf


dict_models = {
    "XR1Cnn": XR1Cnn,
    "MR1CnnTrf": MR1CnnTrf,
    "MR2CnnTrf": MR2CnnTrf,
    "XR1MR1CnnTrf": XR1MR1CnnTrf,
    "XR1MR2CnnTrf": XR1MR2CnnTrf,
    "XR1MR2C1CnnTrf": XR1MR2C1CnnTrf,
}
