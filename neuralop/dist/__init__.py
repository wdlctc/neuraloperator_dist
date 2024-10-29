from .darcy import load_darcy_flow_small
from ..datasets import darcy 

from .fno_block import forward_with_postactivation
from ..layers.fno_block import FNOBlocks 

from .spectral_convolution import forward
from ..layers.spectral_convolution import SpectralConv 

def set_image_parallel():
    darcy.load_darcy_flow_small = load_darcy_flow_small
    FNOBlocks.forward_with_postactivation = forward_with_postactivation
    SpectralConv.forward = forward