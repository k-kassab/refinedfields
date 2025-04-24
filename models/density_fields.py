"""
The code below has been adapted from K-Planes under the folllowing license:
BSD 3-Clause License

Copyright (c) 2023, "K-Planes for Radiance Fields in Space, Time, and Appearance" authors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 2/24/2023.
"""

from typing import Optional, Callable
import logging as log

import torch
import torch.nn as nn
import tinycudann as tcnn

from models.kplane_field import interpolate_ms_features, normalize_aabb, init_grid_param
from raymarching.spatial_distortions import SpatialDistortion


class KPlaneDensityField(nn.Module):
    def __init__(self,
                 aabb,
                 resolution,
                 num_input_coords,
                 num_output_coords,
                 density_activation: Callable,
                 spatial_distortion: Optional[SpatialDistortion] = None,
                 linear_decoder: bool = True):
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.hexplane = num_input_coords == 4
        self.feature_dim = num_output_coords
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder
        activation = "ReLU"
        if self.linear_decoder:
            activation = "None"

        self.grids = init_grid_param(
            grid_nd=2, in_dim=num_input_coords, out_dim=num_output_coords, reso=resolution,
            a=0.1, b=0.15)
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        print(f"Initialized KPlaneDensityField. hexplane={self.hexplane} - "
                 f"resolution={resolution}")
        log.info(f"Initialized KPlaneDensityField. hexplane={self.hexplane} - "
                 f"resolution={resolution}")
        log.info(f"KPlaneDensityField grids: \n{self.grids}")

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None and self.hexplane:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None)
        density = self.density_activation(
            self.sigma_net(features).to(pts)
            #features.to(pts)
        ).view(n_rays, n_samples, 1)
        return density

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = {k: v for k, v in self.sigma_net.named_parameters(prefix="sigma_net")}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
    