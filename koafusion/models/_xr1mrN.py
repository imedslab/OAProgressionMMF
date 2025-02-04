import math
from collections import OrderedDict
import torch
from torch import nn
from einops import rearrange, reduce, repeat

from ._core_trf import FeaT
from ._core_fes import dict_fes


class XR1MR1CnnTrf(nn.Module):
    def __init__(self, config, path_weights):
        super(XR1MR1CnnTrf, self).__init__()
        self.config = config
        if self.config["debug"]:
            print("Config at model init", self.config)
        self.vs = dict()

        t_fe0 = dict_fes[self.config["fe"]["xr"]["arch"]](
            pretrained=self.config["fe"]["xr"]["pretrained"])
        t_fe1 = dict_fes[self.config["fe"]["mr"]["arch"]](
            pretrained=self.config["fe"]["mr"]["pretrained"])
        if self.config["fe"]["xr"]["with_gap"] or self.config["fe"]["mr"]["with_gap"]:
            # Exclude trailing FC layer
            t_fe0 = list(t_fe0.children())[:-1]
            t_fe1 = list(t_fe1.children())[:-1]
        else:
            # Exclude trailing GAP and FC layers
            t_fe0 = list(t_fe0.children())[:-2]
            t_fe1 = list(t_fe1.children())[:-2]
        self._fe0 = nn.Sequential(*t_fe0)
        self._fe1 = nn.Sequential(*t_fe1)

        if self.config["fe"]["xr"]["dropout"]:
            self._fe0_drop = nn.Dropout2d(p=self.config["fe"]["xr"]["dropout"])
        else:
            self._fe0_drop = nn.Identity()

        if self.config["fe"]["mr"]["dropout"]:
            self._fe1_drop = nn.Dropout2d(p=self.config["fe"]["mr"]["dropout"])
        else:
            self._fe1_drop = nn.Identity()

        if self.config["debug"]:
            print("FE0 submodel", self._fe0)
            print("FE1 submodel", self._fe1)

        mapping = {"resnet18": 512, "resnet34": 512,
                   "resnet50": 2048, "resnext50_32x4d": 2048}
        assert self.config["fe"]["xr"]["arch"] in mapping
        assert self.config["fe"]["mr"]["arch"] in mapping
        self.vs["fe0_out_ch"] = mapping[self.config["fe"]["xr"]["arch"]]
        self.vs["fe1_out_ch"] = mapping[self.config["fe"]["mr"]["arch"]]

        # Calculate FE input and output data shape
        t_0 = self.config["input_size"][0]
        t_1 = self.config["input_size"][1]
        if self.config["downscale"]:
            t_0 = [round(s * d) for s, d in zip(t_0, self.config["downscale"][0])]
            t_1 = [round(s * d) for s, d in zip(t_1, self.config["downscale"][1])]
        self.vs["fe0_shape_in"] = t_0
        self.vs["fe1_shape_in"] = t_1

        mapping = {320: 10, 160: 5, 128: 4, 96: 3, 64: 2, 32: 1,
                   350: 11, 25: 1}
        assert all(e in mapping for e in self.vs["fe0_shape_in"])
        assert all(e in mapping for e in self.vs["fe1_shape_in"][:2])

        if self.config["fe"]["xr"]["with_gap"]:
            self.vs["fe0_out_spat"] = (1, 1)
        else:
            self.vs["fe0_out_spat"] = tuple(mapping[e] for e in self.vs["fe0_shape_in"])

        if self.config["fe"]["mr"]["with_gap"]:
            self.vs["fe1_out_spat"] = (1, 1)
        else:
            self.vs["fe1_out_spat"] = tuple(mapping[e] for e in self.vs["fe1_shape_in"][:2])

        self.vs["agg_in_len_0"] = math.prod(self.vs["fe0_out_spat"])
        self.vs["agg_in_len_1"] = self.config["agg"]["num_slices"][1] * \
                                  math.prod(self.vs["fe1_out_spat"])

        # Using number of channels from MR FE
        self.vs["agg_in_depth"] = self.vs["fe1_out_ch"]

        self._agg = FeaT(
            num_patches=self.vs["agg_in_len_0"] + self.vs["agg_in_len_1"],
            patch_dim=self.vs["agg_in_depth"],
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
        input0 : (B, CH, R#, C#)
        input1 : (B, CH, R#, C#, S#)

        Notes:
            B - batch, CH - channel, R - row, C - column, S - slice/plane, F - feature
        """
        endpoints = OrderedDict()

        shapes0 = input0.size()
        shapes1 = input1.size()
        self._debug_tensor_shape(input0, "input0")
        self._debug_tensor_shape(input1, "input1")

        t_in1 = rearrange(input1, "b ch r c s -> (b s) ch r c")
        t_in0 = repeat(input0, "b ch r c -> b (k ch) r c", k=3)
        t_in1 = repeat(t_in1, "bs ch r c -> bs (k ch) r c", k=3)
        self._debug_tensor_shape(t_in0, "proc in0")
        self._debug_tensor_shape(t_in1, "proc in1")

        res_fe0 = self._fe0(t_in0)
        res_fe1 = self._fe1(t_in1)
        self._debug_tensor_shape(res_fe0, "FE0 out")
        self._debug_tensor_shape(res_fe1, "FE1 out")
        t_fe0 = self._fe0_drop(res_fe0)
        t_fe1 = self._fe1_drop(res_fe1)
        t_fe0 = rearrange(t_fe0, "b ch d0 d1 -> b (d0 d1) ch")
        t_fe1 = rearrange(t_fe1, "(b s) ch d0 d1 -> b (s d0 d1) ch", b=shapes1[0])
        self._debug_tensor_shape(t_fe0, "FE0 proc")
        self._debug_tensor_shape(t_fe1, "FE1 proc")

        t_fe_m = torch.cat([
            t_fe0,
            t_fe1,
        ], dim=1)
        self._debug_tensor_shape(t_fe_m, "t_fe_m")

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


class XR1MR2CnnTrf(nn.Module):
    def __init__(self, config, path_weights):
        super(XR1MR2CnnTrf, self).__init__()
        self.config = config
        if self.config["debug"]:
            print("Config at model init", self.config)
        self.vs = dict()

        t_fe0 = dict_fes[self.config["fe"]["xr"]["arch"]](
            pretrained=self.config["fe"]["xr"]["pretrained"])
        t_fe1 = dict_fes[self.config["fe"]["mr"]["arch"]](
            pretrained=self.config["fe"]["mr"]["pretrained"])
        t_fe2 = dict_fes[self.config["fe"]["mr"]["arch"]](
            pretrained=self.config["fe"]["mr"]["pretrained"])
        if self.config["fe"]["xr"]["with_gap"] or self.config["fe"]["mr"]["with_gap"]:
            # Exclude trailing FC layer
            t_fe0 = list(t_fe0.children())[:-1]
            t_fe1 = list(t_fe1.children())[:-1]
            t_fe2 = list(t_fe2.children())[:-1]
        else:
            # Exclude trailing GAP and FC layers
            t_fe0 = list(t_fe0.children())[:-2]
            t_fe1 = list(t_fe1.children())[:-2]
            t_fe2 = list(t_fe2.children())[:-2]
        self._fe0 = nn.Sequential(*t_fe0)
        self._fe1 = nn.Sequential(*t_fe1)
        self._fe2 = nn.Sequential(*t_fe2)

        if self.config["fe"]["xr"]["dropout"]:
            self._fe0_drop = nn.Dropout2d(p=self.config["fe"]["xr"]["dropout"])
        else:
            self._fe0_drop = nn.Identity()

        if self.config["fe"]["mr"]["dropout"]:
            self._fe1_drop = nn.Dropout2d(p=self.config["fe"]["mr"]["dropout"])
            self._fe2_drop = nn.Dropout2d(p=self.config["fe"]["mr"]["dropout"])
        else:
            self._fe1_drop = nn.Identity()
            self._fe2_drop = nn.Identity()

        if self.config["debug"]:
            print("FE0 submodel", self._fe0)
            print("FE1 submodel", self._fe1)
            print("FE2 submodel", self._fe2)

        mapping = {"resnet18": 512, "resnet34": 512,
                   "resnet50": 2048, "resnext50_32x4d": 2048}
        assert self.config["fe"]["xr"]["arch"] in mapping
        assert self.config["fe"]["mr"]["arch"] in mapping
        self.vs["fe0_out_ch"] = mapping[self.config["fe"]["xr"]["arch"]]
        self.vs["fe12_out_ch"] = mapping[self.config["fe"]["mr"]["arch"]]

        # Calculate FE input and output data shape
        t_0 = self.config["input_size"][0]
        t_1 = self.config["input_size"][1]
        t_2 = self.config["input_size"][2]
        if self.config["downscale"]:
            t_0 = [round(s * d) for s, d in zip(t_0, self.config["downscale"][0])]
            t_1 = [round(s * d) for s, d in zip(t_1, self.config["downscale"][1])]
            t_2 = [round(s * d) for s, d in zip(t_2, self.config["downscale"][2])]
        self.vs["fe0_shape_in"] = t_0
        self.vs["fe1_shape_in"] = t_1
        self.vs["fe2_shape_in"] = t_2

        mapping = {320: 10, 160: 5, 128: 4, 96: 3, 64: 2, 32: 1,
                   350: 11, 25: 1}
        assert all(e in mapping for e in self.vs["fe0_shape_in"])
        assert all(e in mapping for e in self.vs["fe1_shape_in"][:2])
        assert all(e in mapping for e in self.vs["fe2_shape_in"][:2])

        if self.config["fe"]["xr"]["with_gap"]:
            self.vs["fe0_out_spat"] = (1, 1)
        else:
            self.vs["fe0_out_spat"] = tuple(mapping[e] for e in self.vs["fe0_shape_in"])

        if self.config["fe"]["mr"]["with_gap"]:
            self.vs["fe1_out_spat"] = (1, 1)
            self.vs["fe2_out_spat"] = (1, 1)
        else:
            self.vs["fe1_out_spat"] = tuple(mapping[e] for e in self.vs["fe1_shape_in"][:2])
            self.vs["fe2_out_spat"] = tuple(mapping[e] for e in self.vs["fe2_shape_in"][:2])

        self.vs["agg_in_len_0"] = math.prod(self.vs["fe0_out_spat"])
        self.vs["agg_in_len_1"] = self.config["agg"]["num_slices"][1] * \
                                  math.prod(self.vs["fe1_out_spat"])
        self.vs["agg_in_len_2"] = self.config["agg"]["num_slices"][2] * \
                                  math.prod(self.vs["fe2_out_spat"])

        # Using number of channels from MR FEs
        self.vs["agg_in_depth"] = self.vs["fe12_out_ch"]

        self._agg_1 = FeaT(
            num_patches=self.vs["agg_in_len_1"],
            patch_dim=self.vs["agg_in_depth"],
            emb_dim=self.vs["agg_in_depth"],
            depth=self.config["agg"]["depth"],
            heads=self.config["agg"]["heads"],
            mlp_dim=self.config["agg"]["mlp_dim"],
            num_classes=self.config["output_channels"],
            emb_dropout=self.config["agg"]["emb_dropout"],
            with_cls=False,
            # num_cls_tokens=1,
            mlp_dropout=self.config["agg"]["mlp_dropout"],
            # num_outputs=1,
        )
        self._agg_2 = FeaT(
            num_patches=self.vs["agg_in_len_2"],
            patch_dim=self.vs["agg_in_depth"],
            emb_dim=self.vs["agg_in_depth"],
            depth=self.config["agg"]["depth"],
            heads=self.config["agg"]["heads"],
            mlp_dim=self.config["agg"]["mlp_dim"],
            num_classes=self.config["output_channels"],
            emb_dropout=self.config["agg"]["emb_dropout"],
            with_cls=False,
            # num_cls_tokens=1,
            mlp_dropout=self.config["agg"]["mlp_dropout"],
            # num_outputs=1,
        )
        self._agg_final = FeaT(
            num_patches=self.vs["agg_in_len_0"] + self.vs["agg_in_len_1"] + self.vs["agg_in_len_2"],
            patch_dim=self.vs["agg_in_depth"],
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

    def forward(self, input0, input1, input2):
        """
        input0 : (B, CH, R#, C#)
        input1 : (B, CH, R#, C#, S#)
        input2 : (B, CH, R#, C#, S#)

        Notes:
            B - batch, CH - channel, R - row, C - column, S - slice/plane, F - feature
        """
        endpoints = OrderedDict()

        shapes0 = input0.size()
        shapes1 = input1.size()
        shapes2 = input2.size()
        self._debug_tensor_shape(input0, "input0")
        self._debug_tensor_shape(input1, "input1")
        self._debug_tensor_shape(input2, "input2")

        t_in1 = rearrange(input1, "b ch r c s -> (b s) ch r c")
        t_in2 = rearrange(input2, "b ch r c s -> (b s) ch r c")
        t_in0 = repeat(input0, "b ch r c -> b (k ch) r c", k=3)
        t_in1 = repeat(t_in1, "bs ch r c -> bs (k ch) r c", k=3)
        t_in2 = repeat(t_in2, "bs ch r c -> bs (k ch) r c", k=3)
        self._debug_tensor_shape(t_in0, "proc in0")
        self._debug_tensor_shape(t_in1, "proc in1")
        self._debug_tensor_shape(t_in2, "proc in2")

        res_fe0 = self._fe0(t_in0)
        res_fe1 = self._fe1(t_in1)
        res_fe2 = self._fe2(t_in2)
        self._debug_tensor_shape(res_fe0, "FE0 out")
        self._debug_tensor_shape(res_fe1, "FE1 out")
        self._debug_tensor_shape(res_fe2, "FE2 out")
        t_fe0 = self._fe0_drop(res_fe0)
        t_fe1 = self._fe1_drop(res_fe1)
        t_fe2 = self._fe2_drop(res_fe2)
        t_fe0 = rearrange(t_fe0, "b ch d0 d1 -> b (d0 d1) ch")
        t_fe1 = rearrange(t_fe1, "(b s) ch d0 d1 -> b (s d0 d1) ch", b=shapes1[0])
        t_fe2 = rearrange(t_fe2, "(b s) ch d0 d1 -> b (s d0 d1) ch", b=shapes2[0])
        self._debug_tensor_shape(t_fe0, "FE0 proc")
        self._debug_tensor_shape(t_fe1, "FE1 proc")
        self._debug_tensor_shape(t_fe2, "FE2 proc")

        _, res_agg1, _ = self._agg_1(t_fe1)
        _, res_agg2, _ = self._agg_2(t_fe2)
        self._debug_tensor_shape(res_agg1, "AGG1 out")
        self._debug_tensor_shape(res_agg2, "AGG2 out")

        t_fe_m = torch.cat([
            t_fe0,
            res_agg1,
            res_agg2,
        ], dim=1)
        self._debug_tensor_shape(t_fe_m, "t_fe_m")

        res_agg_final, _, _ = self._agg_final(t_fe_m)
        self._debug_tensor_shape(res_agg_final, "AGG_FINAL out")

        res_out = rearrange(res_agg_final, "b head cls -> b (head cls)")

        endpoints["main"] = res_out

        if self.config.output_type == "main":
            return endpoints["main"]
        elif self.config.output_type == "dict":
            return endpoints
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")
