import re
from dataclasses import dataclass
from typing import Tuple, Union

import torch.nn as nn

NumberOrTuple = Union[int, Tuple[int, ...]]


# --------------------------------------------------------------------------- #
# 1.  Dataclass that *represents* the stitching convolution                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConvSpec:
    out_channels: int
    kernel_size: NumberOrTuple
    stride: NumberOrTuple = 1
    padding: NumberOrTuple = 0

    def build(
        self,
        in_channels: int,
        bias: bool = True,
        groups: int = 1,
    ) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode="replicate",
            groups=groups,
            bias=bias,
        )


# --------------------------------------------------------------------------- #
# 2.  Regex-based parser                                                      #
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"^conv2d_"
    r"k(?P<k>[0-9x]+)_"  # kernel size
    r"o(?P<o>[0-9]+)"  # out channels (REQUIRED)
    r"(?:_s(?P<s>[0-9x]+))?"  # optional stride
    r"(?:_p(?P<p>[0-9x]+))?"  # optional padding
    r"$",
    re.IGNORECASE,
)


def _to_int_or_tuple(txt: str | None) -> NumberOrTuple:
    if not txt:
        # caller handles defaults
        return 0
    if "x" in txt:
        return tuple(int(n) for n in txt.split("x"))
    return int(txt)


class ResidualStitchingLayer(nn.Module):
    """Zero-initialized residual branch on top of the closed-form stitching conv.

    output = F(x) + G(F(x))

    where F is the fitted stitching conv and G is a small bottleneck MLP
    (SiLU -> 1x1 conv -> SiLU -> 1x1 conv) whose last layer is zero-initialized,
    so at init G(.) = 0 and the output equals F(x). During training, G learns
    nonlinear residual corrections.
    """

    def __init__(self, stitching_conv: nn.Conv2d):
        super().__init__()
        self.stitching_conv = stitching_conv
        out_ch = stitching_conv.out_channels
        hidden = max(1, out_ch // 8)
        self.residual = nn.Sequential(
            nn.SiLU(), nn.Conv2d(out_ch, hidden, kernel_size=1),
            nn.SiLU(), nn.Conv2d(hidden, out_ch, kernel_size=1),
        )
        # Zero-init the last conv so G(.) = 0 at init
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        fx = self.stitching_conv(x)
        return fx + self.residual(fx)


def parse_conv_spec(spec: str) -> ConvSpec:
    """
    Convert 'conv2d_k1x1_o1280_s1x1_p0x0' -> ConvSpec(...)
    Raises ValueError if the string doesn't follow the grammar.
    """
    m = _TOKEN_RE.fullmatch(spec)
    if not m:
        raise ValueError(
            f"Bad CONV_SPEC {spec!r}. Expected something like "
            "'conv2d_k1x1_o1280_s1x1_p0x0'."
        )

    g = m.groupdict()
    return ConvSpec(
        out_channels=int(g["o"]),
        kernel_size=_to_int_or_tuple(g["k"]),
        stride=_to_int_or_tuple(g["s"]) if g["s"] else 1,
        padding=_to_int_or_tuple(g["p"]) if g["p"] else 0,
    )
