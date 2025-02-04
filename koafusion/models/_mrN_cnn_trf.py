import math
import copy
from collections import OrderedDict
import torch
from torch import nn
from einops import rearrange, reduce, repeat

from ._core_trf import FeaT
from ._core_fes import dict_fes


class MR1CnnTrf(nn.Module):
    def __init__(self, config, path_weights):
        super(MR1CnnTrf, self).__init__()
        self.config = config
        if self.config["debug"]:
            print("Config at model init", self.config)
        self.vs = dict()

        t_fe = dict_fes[self.config["fe"]["arch"]](
            pretrained=self.config["fe"]["pretrained"])
        if self.config["fe"]["with_gap"]:
            # Exclude trailing FC layer
            t_fe = list(t_fe.children())[:-1]
        else:
            # Exclude trailing GAP and FC layers
            t_fe = list(t_fe.children())[:-2]
        self._fe = nn.Sequential(*t_fe)

        if self.config["fe"]["dropout"]:
            self._fe_drop = nn.Dropout2d(p=self.config["fe"]["dropout"])
        else:
            self._fe_drop = nn.Identity()

        if self.config["debug"]:
            print("FE submodel", self._fe)

        if self.config["fe"]["arch"] in ("resnet18", "resnet34"):
            self.vs["fe_out_ch"] = 512
        elif self.config["fe"]["arch"] == "resnet50":
            self.vs["fe_out_ch"] = 2048
        else:
            raise ValueError(f"Unsupported `model.fe.arch`")

        # Calculate input data shape
        t = self.config["input_size"][0]
        if self.config["downscale"]:
            t = [round(s * d) for s, d in zip(t, self.config["downscale"][0])]
        self.vs["shape_in"] = t

        if self.config["fe"]["with_gap"]:
            self.vs["fe_out_spat"] = (1, 1, 1)
        else:
            try:
                mapping = {320: 10, 160: 5, 128: 4, 96: 3, 64: 2, 32: 1}
                self.vs["fe_out_spat"] = tuple(mapping[e] for e in self.vs["shape_in"])
            except (ValueError, IndexError) as _:
                msg = "Unspecified `model.fe` output shape for given `model.input_size`"
                raise ValueError(msg)

        if self.config["fe"]["dims_view"] == "rc":
            self.vs["agg_in_len"] = self.vs["shape_in"][2] * \
                                    (self.vs["fe_out_spat"][0] * self.vs["fe_out_spat"][1])
        elif self.config["fe"]["dims_view"] == "cs":
            self.vs["agg_in_len"] = self.vs["shape_in"][0] * \
                                    (self.vs["fe_out_spat"][1] * self.vs["fe_out_spat"][2])
        elif self.config["fe"]["dims_view"] == "rs":
            self.vs["agg_in_len"] = self.vs["shape_in"][1] * \
                                    (self.vs["fe_out_spat"][0] * self.vs["fe_out_spat"][2])
        else:
            raise ValueError(f"Unsupported `model.fe.dims_view`")

        self.vs["agg_in_depth"] = self.vs["fe_out_ch"]

        self._agg = FeaT(
            num_patches=self.vs["agg_in_len"],
            patch_dim=self.vs["agg_in_depth"],
            # emb_dim=self.config["agg"]["emb_dim"],
            emb_dim=self.vs["agg_in_depth"],
            depth=self.config["agg"]["depth"],
            heads=self.config["agg"]["heads"],
            mlp_dim=self.config["agg"]["mlp_dim"],
            num_classes=self.config["output_channels"],
            emb_dropout=self.config["agg"]["emb_dropout"],
            # with_cls=True,
            # num_cls_tokens=1,
            mlp_dropout=self.config["agg"]["mlp_dropout"],
            # num_outputs=1,
        )

        if self.config["restore_weights"]:
            self.load_state_dict(torch.load(path_weights))

    def _debug_tensor_shape(self, tensor, name=""):
        if self.config["debug"]:
            print(f"Shape of {name} is", tensor.size())

    def forward(self, input):
        """
        input : (B, CH, R, C, S)

        Notes:
            B - batch, CH - channel, R - row, C - column, S - slice/plane, F - feature
        """
        endpoints = OrderedDict()

        shapes = input.size()
        self._debug_tensor_shape(input, "input")

        t_in = repeat(input, "b ch r c s -> b (k ch) r c s", k=3)

        if self.config["fe"]["dims_view"] == "rc":
            t_in = rearrange(t_in, "b ch r c s -> (b s) ch r c")
        elif self.config["fe"]["dims_view"] == "cs":
            t_in = rearrange(t_in, "b ch r c s -> (b r) ch c s")
        elif self.config["fe"]["dims_view"] == "rs":
            t_in = rearrange(t_in, "b ch r c s -> (b c) ch r s")

        self._debug_tensor_shape(t_in, "proc in")

        res_fe = self._fe(t_in)
        self._debug_tensor_shape(res_fe, "FE out")
        t_fe = self._fe_drop(res_fe)
        t_fe = rearrange(t_fe, "(b d2) ch d0 d1 -> b (d2 d0 d1) ch", b=shapes[0])
        self._debug_tensor_shape(t_fe, "FE proc")

        res_agg, _, _ = self._agg(t_fe)
        self._debug_tensor_shape(res_agg, "AGG out")

        res_out = rearrange(res_agg, "b head cls -> b (head cls)")

        endpoints["main"] = res_out

        if self.config.output_type == "main":
            return endpoints["main"]
        elif self.config.output_type == "dict":
            return endpoints
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")


class MR2CnnTrf(nn.Module):
    def __init__(self, config, path_weights):
        super(MR2CnnTrf, self).__init__()
        self.config = config
        if self.config["debug"]:
            print("Config at model init", self.config)
        self.vs = dict()

        t_fe0 = copy.deepcopy(dict_fes[self.config["fe"]["arch"]](
            pretrained=self.config["fe"]["pretrained"]))
        t_fe1 = copy.deepcopy(dict_fes[self.config["fe"]["arch"]](
            pretrained=self.config["fe"]["pretrained"]))
        if self.config["fe"]["with_gap"]:
            # Exclude trailing FC layer
            t_fe0 = list(t_fe0.children())[:-1]
            t_fe1 = list(t_fe1.children())[:-1]
        else:
            # Exclude trailing GAP and FC layers
            t_fe0 = list(t_fe0.children())[:-2]
            t_fe1 = list(t_fe1.children())[:-2]
        self._fe0 = nn.Sequential(*t_fe0)
        self._fe1 = nn.Sequential(*t_fe1)

        if self.config["fe"]["dropout"]:
            self._fe0_drop = nn.Dropout2d(p=self.config["fe"]["dropout"])
            self._fe1_drop = nn.Dropout2d(p=self.config["fe"]["dropout"])
        else:
            self._fe0_drop = nn.Identity()
            self._fe1_drop = nn.Identity()

        if self.config["debug"]:
            print("FE0 submodel", self._fe0)
            print("FE1 submodel", self._fe1)

        if self.config["fe"]["arch"] in ("resnet18", "resnet34"):
            self.vs["fe_out_ch"] = 512
        elif self.config["fe"]["arch"] == "resnet50":
            self.vs["fe_out_ch"] = 2048
        else:
            raise ValueError(f"Unsupported `model.fe.arch`")

        if self.config["fe"]["with_gap"]:
            self.vs["fe_out_spat"] = (1, 1)
        else:
            if self.config["input_size"][0][0] == 320:
                self.vs["fe_out_spat"] = (5, 5)
            else:
                msg = "Unspecified `model.fe` output shape for given `model.input_size`"
                raise ValueError(msg)

        self.vs["agg_in_len"] = (self.config["agg"]["num_slices"][0] +
                                 self.config["agg"]["num_slices"][1]) * \
                                 math.prod(self.vs["fe_out_spat"])
        self.vs["agg_in_depth"] = self.vs["fe_out_ch"]

        # self._ups = nn.Upsample(scale_factor=(2, 1), mode="nearest")

        self._agg = FeaT(
            num_patches=self.vs["agg_in_len"],
            patch_dim=self.vs["agg_in_depth"],
            # emb_dim=self.config["agg"]["emb_dim"],
            emb_dim=self.vs["agg_in_depth"],
            depth=self.config["agg"]["depth"],
            heads=self.config["agg"]["heads"],
            mlp_dim=self.config["agg"]["mlp_dim"],
            num_classes=self.config["output_channels"],
            emb_dropout=self.config["agg"]["emb_dropout"],
            # with_cls=True,
            # num_cls_tokens=1,
            mlp_dropout=self.config["agg"]["mlp_dropout"],
            # num_outputs=1,
        )

        if self.config["restore_weights"]:
            self.load_state_dict(torch.load(path_weights))

    def _debug_tensor_shape(self, tensor, name=""):
        if self.config["debug"]:
            print(f"Shape of {name} is", tensor.size())

    def forward(self, input0, input1):
        """
        input0 : (B, CH, R#, C#, S#)
        input1 : (B, CH, R#, C#, S#)

        Notes:
            B - batch, CH - channel, R - row, C - column, S - slice/plane, F - feature
        """
        endpoints = OrderedDict()

        shapes0 = input0.size()
        shapes1 = input1.size()
        self._debug_tensor_shape(input0, "input0")
        self._debug_tensor_shape(input1, "input1")

        t_in0 = rearrange(input0, "b ch r c s -> (b s) ch r c")
        t_in1 = rearrange(input1, "b ch r c s -> (b s) ch r c")
        t_in0 = repeat(t_in0, "bs ch r c -> bs (k ch) r c", k=3)
        t_in1 = repeat(t_in1, "bs ch r c -> bs (k ch) r c", k=3)
        self._debug_tensor_shape(t_in0, "proc in0")
        self._debug_tensor_shape(t_in1, "proc in1")

        res_fe0 = self._fe0(t_in0)
        res_fe1 = self._fe1(t_in1)
        self._debug_tensor_shape(res_fe0, "FE0 out")
        self._debug_tensor_shape(res_fe1, "FE1 out")
        t_fe0 = self._fe0_drop(res_fe0)
        t_fe1 = self._fe1_drop(res_fe1)
        t_fe0 = rearrange(t_fe0, "(b s) ch d0 d1 -> b (s d0 d1) ch", b=shapes0[0])
        t_fe1 = rearrange(t_fe1, "(b s) ch d0 d1 -> b (s d0 d1) ch", b=shapes1[0])
        self._debug_tensor_shape(t_fe0, "FE0 proc")
        self._debug_tensor_shape(t_fe1, "FE1 proc")

        t_fe_m = torch.cat([
            t_fe0,
            t_fe1,
        ], dim=1)

        res_agg, _, _ = self._agg(t_fe_m)
        self._debug_tensor_shape(res_agg, "AGG out")

        res_out = rearrange(res_agg, "b head cls -> b (head cls)")

        endpoints["main"] = res_out

        if self.config.output_type == "main":
            return endpoints["main"]
        elif self.config.output_type == "dict":
            return endpoints
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")
