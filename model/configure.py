import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from .Utils import *
import random

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)