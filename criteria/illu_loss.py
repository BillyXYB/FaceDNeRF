import sys
sys.path.append('../DPR_model')
sys.path.append('../DPR_utils')
import kornia
from utils_SH import *
from defineHourglass_512_gray_skip import *
import numpy as np
import torch

def illu_loss(relight_model,sh,synth_images,device):  ####### input synth_images range: [1,3,512,512
    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = torch.from_numpy(sh).to(device)
    sh.requires_grad = False
    Lab = kornia.color.rgb_to_lab(synth_images)
    Lab = Lab.squeeze()
    Lab = torch.permute(Lab, (1,2,0))
    Lab[:,:,0] = Lab[:,:,0] * 2.55
    Lab[:,:,1] = Lab[:,:,1] + 128.0
    Lab[:,:,2] = Lab[:,:,2] + 128.0
    inputL = Lab[:,:,0]/255.0
    inputL = inputL[None,None,...]
    outputImg, outputSH  = relight_model(inputL, sh, 0)
    ill_loss = (outputSH - sh).square().sum()
    return ill_loss
    
    