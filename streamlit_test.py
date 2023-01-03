import keras
import numpy as np
#import efficientnet.keras as efn
import argparse
from PIL import Image
#import streamlit as st
import os

# Fix the mac os error
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Model params
    img_h = 256
    img_w = 256

