import time
import os

#### Runing PTI neads two 3090/4090 GPUs at least. ####
gpu_id1 = 0
gpu_id2 = 1 
num_steps = 1000
num_steps_pti = 400

#### For the hyper-parameters, please refer to the paper to adjust them. Generally, a higher value means a higher stength of the corresponding control. ####
image_ids = ["00018"]  
input_dict = [{"text": "A woman wearing a pair of glasses", "lamda_id": 0.4, "lamda_origin": 0.4,  "lamda_diffusion": 2.1e-5, "lamda_illumination":0.0},
              {"text": "A woman wearing a pair of glasses", "lamda_id": 0.4, "lamda_origin": 0.4,  "lamda_diffusion": 2.1e-5, "lamda_illumination":0.01},
              {"text": "", "lamda_id": 0.4, "lamda_origin": 0.4,  "lamda_diffusion": 0.0, "lamda_illumination":0.01},
              {"text": "A young girl is about 15 years old", "lamda_id": 0.4, "lamda_origin": 0.4,  "lamda_diffusion": 2.1e-5, "lamda_illumination":0.0},
              {"text": "A old lady is about 60 years old", "lamda_id": 0.4, "lamda_origin": 0.4,  "lamda_diffusion": 2.1e-5, "lamda_illumination":0.0},
              ]
# image_ids = ["00064"]  
# input_dict = [{"text": "A boy is about ten years old", "lamda_id": 0.4, "lamda_origin": 0.4,  "lamda_diffusion": 2.1e-5, "lamda_illumination":0, "lamda_latent_regu":0},
#               {"text": "A man with short blue hair", "lamda_id": 0.4, "lamda_origin": 0.4,  "lamda_diffusion": 2.1e-5, "lamda_illumination":0, "lamda_latent_regu":0},
#               ]
output_dir = "./output/"

#####
for image_id in image_ids:
    for i in range(len(input_dict)):
        print("Start : %s" % time.ctime())
        # output_dir = "projector_out/" + image_id +"_supp_ablation/" + image_id + "_" + str(input_dict[i]["lamda_id"]) + "_" + str(input_dict[i]["lamda_origin"]) + "_" + str(input_dict[i]["lamda_diffusion"]) + "_" + str(input_dict[i]["lamda_latent_regu"])
        print(output_dir)
        command = "CUDA_VISIBLE_DEVICES=" + str(gpu_id1)+","+str(gpu_id2) + " python run.py --outdir='"+ output_dir + "' --network=/disk1/haozhang/EG3D-diffusion_zh/eg3d/networks/ffhqrebalanced512-128.pkl --sample_mult=2 \
            --image_path ./test_data/" + image_id + ".png --c_path ./test_data/" + image_id+ ".npy "+ " --num_steps "+str(num_steps)+" --num_steps_pti "+str(num_steps_pti)+" --description '"+ input_dict[i]["text"]+"' \
                --lamda_id " + str(input_dict[i]["lamda_id"]) + " --lamda_origin " + str(input_dict[i]["lamda_origin"]) + " --lamda_diffusion " + str(input_dict[i]["lamda_diffusion"]) + " --lamda_illumination " + str(input_dict[i]["lamda_illumination"]) 
        print(command)
        os.system(command)
        # time.sleep(600)
        print("End : %s" % time.ctime())
        ### generate result video
        output_dir_video = os.path.join(output_dir, image_id+input_dict[i]["text"]+str(input_dict[i]["lamda_id"])+" "+str(input_dict[i]["lamda_origin"])+" "+str(input_dict[i]["lamda_diffusion"])+" "+str(input_dict[i]["lamda_illumination"]))
        command = "CUDA_VISIBLE_DEVICES=" + str(gpu_id1)+" python gen_videos_from_given_latent_code.py --outdir='"+output_dir_video+"' --trunc=0.7 --npy_path '"+output_dir_video+"/checkpoints/"+image_id+".npy'   --network='"+output_dir_video+"/checkpoints/fintuned_generator.pkl' --sample_mult=2"
        print(command)
        os.system(command)
