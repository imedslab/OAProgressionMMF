from collections import OrderedDict
import torch
from torch import nn
from einops import rearrange, reduce, repeat

from ._core_fes import dict_fes


class XR1Cnn(nn.Module):
    def __init__(self, config, path_weights):
        super(XR1Cnn, self).__init__()
        self.config = config
        if self.config["debug"]:
            print("Config at model init", self.config)

        self._fe = dict_fes[self.config["fe"]["arch"]](
            pretrained=self.config["fe"]["pretrained"])
        # Initialize without the trailing FC layer
        self._fe = nn.Sequential(*list(self._fe.children())[:-1])
        if self.config["debug"]:
            print("FE submodel", self._fe)

        if self.config["fe"]["arch"] in ("resnet18", "resnet34"):
            num_elems = 512
        elif self.config["fe"]["arch"] in ("resnet50", "resnext50_32x4d"):
            num_elems = 2048
        else:
            msg = f"Unknown `num_elems` for `model.fe` output. Get via `model.debug=true`"
            raise ValueError(msg)

        self._agg = nn.Sequential(*[
            nn.Dropout(self.config["agg"]["dropout"]),
            nn.Linear(num_elems, self.config["agg"]["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(self.config["agg"]["dropout"]),
        ])

        self._final = nn.Linear(self.config["agg"]["hidden_size"],
                                self.config["output_channels"])

        if self.config["restore_weights"]:
            self.load_state_dict(torch.load(path_weights))

    def _debug_tensor_shape(self, tensor, name=""):
        if self.config["debug"]:
            print(f"Shape of {name} is", tensor.size())

    def forward(self, input):
        """
        input : (B, CH, R, C)

        Notes:
            B - batch, CH - channel, R - row, C - column, F - feature
        """
        endpoints = OrderedDict()

        shapes = input.size()
        self._debug_tensor_shape(input, "input")

        tmp_in = repeat(input, "b ch r c -> b (k ch) r c", k=3)
        self._debug_tensor_shape(tmp_in, "proc in")

        res_fe = self._fe(tmp_in)
        self._debug_tensor_shape(res_fe, "FE out")

        tmp_fe = rearrange(res_fe, "b ch d0 d1 -> b (ch d0 d1)")
        self._debug_tensor_shape(tmp_fe, "FE proc")

        res_agg = self._agg(tmp_fe)
        self._debug_tensor_shape(res_agg, "AGG out")

        res_out = self._final(res_agg)

        endpoints["main"] = res_out

        if self.config.output_type == "main":
            return endpoints["main"]
        elif self.config.output_type == "dict":
            return endpoints
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")
