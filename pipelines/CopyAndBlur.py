import torch
import numpy as np

from pipelines.BackgroundCopy import BackgroundCopy
from pipelines.MaskBlur import MaskBlur

# Everything is implemented by the parents of this class.
class CopyAndBlur(BackgroundCopy, MaskBlur):
