 # Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fu
from tqdm import tqdm
import dnnlib
import PIL
#import clip
import cv2
from camera_utils import LookAtPoseSampler
import kornia
import logging
import math
from editors.sh import SH_project_polar_function 

# def getLightIntensity(theta, phi):
#     intensity = max(0.0, -4 * math.sin((theta - math.pi/3.0) - math.pi) * math.cos(phi - math.pi/2.0) - 3)
#     return intensity

# def getLightIntensity(theta, phi):
#     intensity = max(0.0, -4 * math.sin(theta - math.pi) * math.cos(phi - 0.6) - 3)
#     return intensity

def getLightIntensity(theta, phi):
    intensity = max(0.0, -4 * math.sin(theta - math.pi) * math.cos(phi - 2.5) - 3)
    return intensity

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


def project(
        G,
        c,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e3,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=1,
        w_name: str,
        text_prompt:str,
        relight_model,
        illu_loss,
        id_loss,
        guidance,
        lamda_id,
        lamda_origin,
        lamda_diffusion,
        lamda_illumination,        
):
    outdir = os.path.join(outdir, "pre")
    os.makedirs(outdir, exist_ok=True)
    log_out_dir = os.path.join(outdir, "log.txt")
    
    logging.basicConfig(filename=log_out_dir, level=logging.INFO)
    #################################
    #the hyperparameters: weight of i_loss, weight of d_loss, and weight of original_loss
    
    weight_of_i_loss = lamda_id
    weight_of_d_loss = lamda_diffusion
    weight_of_original_loss = lamda_origin
    weight_of_illu_loss = lamda_illumination
    
    logging.info("weight_of_i_loss: "+str(weight_of_i_loss))
    logging.info("weight_of_d_loss: "+str(weight_of_d_loss))
    logging.info("weight_of_original_loss: "+str(weight_of_original_loss))
    logging.info("weight_of_illu_loss: "+str(weight_of_illu_loss))

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

    for p in guidance.parameters():
        p.requires_grad = False
    
    for p in relight_model.parameters():
        p.requires_grad = False    
    w_avg_path = './w_avg.npy'
    w_std_path = './w_std.npy'
    if (not os.path.exists(w_avg_path)) or (not os.path.exists(w_std_path)):
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        focal_length = 4.2647  # FFHQ's FOV
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c_samples = c_samples.repeat(w_avg_samples, 1)

        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_avg_tensor = torch.from_numpy(w_avg).cuda()
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    else:
        raise Exception(' ')

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    #url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    url = './networks/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    target_images_orginal_illu = target_images
    
    if target_images_orginal_illu.shape[2] > 256:
        target_images = F.interpolate(target_images_orginal_illu, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    # target_id_features = 

    start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=0.1)
    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)
        synth_images = G.synthesis(ws,c, noise_mode='const')['image']
        #synth_images size: [1, 3, 512, 512]
            
        if step % image_log_step == 0:
            with torch.no_grad():
                vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_original = (synth_images + 1) * (255 / 2)  # original size is 512
        if synth_images_original.shape[2] > 256:
            synth_images = F.interpolate(synth_images_original, size=(256, 256), mode='area')
        synth_images_relight = F.interpolate(synth_images_original, size=(512, 512), mode='area')
     
        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        original_loss = (target_features - synth_features).square().sum()
        #Diffussion loss and ID loss
        
        text_inputs = guidance.get_text_embeds([text_prompt], [""])
        i_loss = id_loss(synth_images*2/255.0-1,target_images*2/255.0-1)[0]
        
        dist = i_loss*weight_of_i_loss + original_loss*weight_of_original_loss
        logging.info(str(step)+" i_loss: "+str(i_loss.cpu().detach()))
        logging.info(str(step)+" original_loss: "+str(original_loss.cpu().detach()))
        #logging.debug(str(step)+" illu_loss: "+str(ill_loss))
        ############ Other views clip loss:
        #print("camera pose",c.shape)
        
        random_draw = torch.rand(2)
        theta = (random_draw[0]-0.5)*0.4
        phi = (random_draw[1]-0.5)*0.4
        side_cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2 +  theta,
            3.14 / 2 +  phi,
            camera_lookat_point, radius=2.7, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        side_c = torch.cat([side_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) 
        side_synth_images = G.synthesis(ws,side_c, noise_mode='const')['image']
        side_synth_images_print = (side_synth_images + 1) * (255 / 2)  # original size is 512
        side_synth_images = (side_synth_images + 1) /2.0
        side_synth_images = F.interpolate(side_synth_images, size=(512, 512), mode='area')
        ################print out the sides views
        vis_img = side_synth_images_print.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')
        #################
        #diffusion loss
        d_loss,real_d_loss = guidance.train_step(text_inputs, side_synth_images, as_latent=False)
        logging.info(str(step)+" d_loss*0.001: "+str(f'{real_d_loss.cpu().detach():.20f}'))
        dist = dist + d_loss[0] * weight_of_d_loss
                
        ######################
        # illu loss
        sh = SH_project_polar_function(getLightIntensity,phi,theta)
        sh = np.array(sh) * 2.5
        ill_loss = illu_loss(relight_model,sh,side_synth_images,torch.device('cuda')) #input sh (type numpy)
        logging.info(str(step)+" illu loss: "+str(f'{ill_loss.cpu().detach():.20f}'))
        
        dist = dist + ill_loss * weight_of_illu_loss
        
        ######################
        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm(w_opt, 1)
        optimizer.step()
        
        #logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        torch.cuda.empty_cache()

    del G
    #torch.cuda.empty_cache()
    return w_opt


def project_pti(
        G, 
        c,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        w_pivot: torch.Tensor,
        *,
        num_steps_pti=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.0003,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e3,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=1,
        w_name: str,
        text_prompt:str,
        relight_model,
        illu_loss,
        id_loss,
        guidance,
        lamda_id,
        lamda_origin,
        lamda_diffusion,
        lamda_illumination,
):
    outdir = os.path.join(outdir, "post")
    os.makedirs(outdir, exist_ok=True)
    log_out_dir = os.path.join(outdir, "log.txt")
    
    logging.basicConfig(filename=log_out_dir, level=logging.INFO)
    #################################
    #the hyperparameters: weight of i_loss, weight of d_loss, and weight of original_loss
    
    weight_of_i_loss = lamda_id
    weight_of_d_loss = lamda_diffusion
    weight_of_original_loss = lamda_origin
    weight_of_illu_loss = lamda_illumination
    
    logging.info("weight_of_i_loss: "+str(weight_of_i_loss))  
    logging.info("weight_of_d_loss: "+str(weight_of_d_loss))
    logging.info("weight_of_original_loss: "+str(weight_of_original_loss))
    logging.info("weight_of_illu_loss: "+str(weight_of_illu_loss))
    ###################################
    

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore
    w_pivot = w_pivot.to(device).detach()
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999),
                                 lr=initial_learning_rate)
    for p in guidance.parameters():
        p.requires_grad = False
    
    for p in relight_model.parameters():
        p.requires_grad = False    
    ##set the requires_grad of guidance to false
    ##
    # Compute w stats.
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647  # FFHQ's FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    #url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    url = './networks/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    target_images_orginal_illu = target_images
   
    if target_images_orginal_illu.shape[2] > 256:
        target_images = F.interpolate(target_images_orginal_illu, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True).to(torch.device('cuda:1'))
    vgg16 = vgg16.to(torch.device('cuda:1'))
    id_loss = id_loss.to(torch.device('cuda:1'))
    target_images = target_images.to(torch.device('cuda:1'))
    torch.cuda.empty_cache()
    # target_id_features = 

    # start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
    # w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
    #                      requires_grad=True)  # pylint: disable=not-callable

    
    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in tqdm(range(num_steps_pti)):
        # Learning rate schedule.
        t = step / num_steps_pti
        #w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # w_noise = torch.randn_like(w_opt) * w_noise_scale
        # ws = (w_opt + w_noise)
        synth_images = G.synthesis(w_pivot,c, noise_mode='const')['image'].to(torch.device('cuda:1'))
        #synth_images size: [1, 3, 512, 512]
            
        if step % image_log_step == 0:
            with torch.no_grad():
                vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_original = (synth_images + 1) * (255 / 2)  # original size is 512
        if synth_images_original.shape[2] > 256:
            synth_images = F.interpolate(synth_images_original, size=(256, 256), mode='area')
        #synth_images_relight = F.interpolate(synth_images_original, size=(512, 512), mode='area')
     
        # Features for synth images.
        
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        original_loss = (target_features - synth_features).square().sum()
        #Diffussion loss and ID loss
        
        i_loss = id_loss(synth_images*2/255.0-1,target_images*2/255.0-1)[0]
        dist = original_loss*weight_of_original_loss +  i_loss*weight_of_i_loss 
        logging.info(str(step)+" i_loss: "+str(i_loss.cpu().detach()))
        logging.info(str(step)+" original_loss: "+str(original_loss.cpu().detach()))
 
        ############ Other views clip loss:
        #print("camera pose",c.shape)
        torch.cuda.empty_cache()
        
        random_draw = torch.rand(2)
        theta = (random_draw[0]-0.5)*0.4
        phi = (random_draw[1]-0.5)*0.4
        side_cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2 +  theta,
            3.14 / 2 +  phi,
            camera_lookat_point, radius=2.7, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        side_c = torch.cat([side_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        # G =G.to(torch.device('cuda:1'))
        # w_pivot = w_pivot.to(torch.device('cuda:1'))
        side_synth_images = G.synthesis(w_pivot,side_c, noise_mode='const')['image'].to(torch.device('cuda:1'))
        side_synth_images_print = (side_synth_images + 1) * (255 / 2)  # original size is 512
        side_synth_images = (side_synth_images + 1) /2.0
        side_synth_images = F.interpolate(side_synth_images, size=(512, 512), mode='area')
        # ################print out the sides views
        vis_img = side_synth_images_print.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')
        #################
        #diffusion loss
        guidance = guidance.to(torch.device('cuda:1'))
        text_inputs = guidance.get_text_embeds([text_prompt], [""])
        d_loss,real_d_loss = guidance.train_step(text_inputs, side_synth_images, as_latent=False)
        logging.info(str(step)+" d_loss: "+str(f'{real_d_loss.cpu().detach():.20f}'))
        dist = dist.to(torch.device('cuda:1')) + d_loss[0] * weight_of_d_loss
          
        ######################
        # illu loss
        sh = SH_project_polar_function(getLightIntensity,phi,theta)
        sh = np.array(sh) * 2.5
        ill_loss = illu_loss(relight_model,sh,side_synth_images,torch.device('cuda:1')) #input sh (type numpy)
        logging.info(str(step)+" illu loss: "+str(f'{ill_loss.cpu().detach():.20f}'))
        
        dist = dist + ill_loss * weight_of_illu_loss
            
        ######################
        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :].to(torch.device('cuda:1'))  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm(w_opt, 1)
        optimizer.step()
        
        #logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        torch.cuda.empty_cache()
    return G
