from typing import Tuple, Optional

import torch
import torch.nn as nn


# based on https://github.com/yenchenlin/nerf-pytorch/
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)   # identity function to include first input as it is
            out_dim += d

        N_freqs = self.kwargs['num_freqs']
        max_freq = N_freqs - 1

        # calculate frequency bands based on number of frequency
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        # initialize embedding functions
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# based on https://github.com/yenchenlin/nerf-pytorch/
def get_embedder(multires, log_sampling, periodic_fns, input_dims, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'num_freqs': multires,
        'log_sampling': log_sampling,
        'periodic_fns': periodic_fns,
    }

    return Embedder(**embed_kwargs)


class NeRF(nn.Module):
    def __init__(
        self,
        in_channels,
        N_freqs,
        logscale=True,
        d_input: int = 63,
        n_layers: int = 8,
        d_filter: int = 256,
        skip: Tuple[int] = (3,),
        d_viewdirs: Optional[int] = None
    ):
        super(NeRF, self).__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs
        self.in_channels = in_channels

        self.layers = nn.ModuleList(
            [nn.Linear(d_input, d_filter)] +
            [nn.Linear(d_filter + d_input, d_filter) if i in skip \
                 else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )
        self.funcs = [torch.sin, torch.cos]
        self.cam_embedder = get_embedder(N_freqs, logscale, self.funcs, in_channels)

        """"
        self.N_freqs = N_freqs
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        self.freq_bands = NeRF.init_positional_encoding(logscale, N_freqs)
        """

        if self.d_viewdirs is not None:
            # self.freq_bands_view = NeRF.init_positional_encoding(logscale, self.d_viewdirs)
            self.viewdir_embedder = get_embedder(self.d_viewdirs, logscale, self.funcs, in_channels)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + 27, d_filter // 2)
            self.alpha_out = nn.Linear(d_filter, 1)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            self.output = nn.Linear(d_filter, 4)

    @staticmethod
    def init_positional_encoding(logscale, N_freqs):
        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        return freq_bands

    def apply_positional_encoding(self, x, freq_bands):
        out = []
        for freq in freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return out

    def forward(
        self,
        pts: torch.Tensor,
        viewdirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """"
        Inputs:
        pts: (N_rays * N_samples, 3), 3D query positions
        viewdirs: (N_rays * N_samples, 3), cam view directions
        """

        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

        # out = [cam_positions]
        # out += self.apply_positional_encoding(cam_positions, self.freq_bands)
        # input_ = torch.cat(out, -1)

        # Embed camera position
        input_ = self.cam_embedder(pts)

        # forward camera positions through MLPs
        x = input_
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            # in case of skip connection
            if i in self.skip:
                x = torch.cat([input_, x], dim=-1)

        # to model view dependent effects
        if self.d_viewdirs is not None:
            # Embed view directions
            out = self.viewdir_embedder(viewdirs)

            alpha = self.alpha_out(x)   # density output

            x = self.rgb_filters(x)
            x = torch.cat([x, out], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)
            x = nn.functional.sigmoid(x)    # color output

            x = torch.cat([x, alpha], dim=-1)
        else:
            x = self.output(x)  # color output
        return x
