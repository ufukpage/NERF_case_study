import torch
import torch.nn as nn
from typing import Tuple, Optional


class NeRF(nn.Module):
  def __init__(
    self,
    in_channels, 
    N_freqs,
    logscale=True,
    d_input: int=63,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (3,),
    d_viewdirs: Optional[int] = None
  ):
    super().__init__()
    self.d_input = d_input
    self.skip = skip
    self.act = nn.functional.relu
    self.d_viewdirs = d_viewdirs
    self.N_freqs = N_freqs
    self.in_channels = in_channels
    self.funcs = [torch.sin, torch.cos]
    self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
    
    self.layers = nn.ModuleList(
      [nn.Linear(d_input, d_filter)] +
      [nn.Linear(d_filter + d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    if logscale:
        self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
    else:
        self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)


    if self.d_viewdirs is not None:
      if logscale:
        self.freq_bands_view = 2**torch.linspace(0, self.d_viewdirs-1, self.d_viewdirs)
      else:
        self.freq_bands_view = torch.linspace(1, 2**(self.d_viewdirs-1), self.d_viewdirs)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.branch = nn.Linear(d_filter + 27, d_filter // 2)
      self.alpha_out = nn.Linear(d_filter, 1)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      self.output = nn.Linear(d_filter, 4)
  
  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None
  ) -> torch.Tensor:

    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')
    
    out = [x]
    for freq in self.freq_bands:
        for func in self.funcs:
            out += [func(freq*x)]

    input_ = torch.cat(out, -1)
    x = input_
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([input_, x], dim=-1)
      
    if self.d_viewdirs is not None:
      out = [viewdirs]
      for freq in self.freq_bands_view:
        for func in self.funcs:
          out += [func(freq*viewdirs)]

      out = torch.cat(out, -1)
      alpha = self.alpha_out(x)
      x = self.rgb_filters(x)
      x = torch.cat([x, out], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)
      x = nn.functional.sigmoid(x)

      x = torch.cat([x, alpha], dim=-1)
    else:
      x = self.output(x)
    return x
