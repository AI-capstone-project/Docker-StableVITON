from preprocess.detectron2.projects.DensePose.apply_net_gradio import DensePose4Gradio
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

import os
import sys
import time
from glob import glob
from os.path import join as opj
from pathlib import Path

import gradio as gr
import torch
from omegaconf import OmegaConf
from PIL import Image
import spaces
print(torch.cuda.is_available(), torch.cuda.device_count())


from cldm.model import create_model
from cldm.plms_hacked import PLMSSampler
from utils_stableviton import get_mask_location, get_batch, tensor2img, center_crop

import aiohttp
import asyncio

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

IMG_H = 1024//2
IMG_W = 768//2

ID = 1

openpose_model_hd = OpenPose(0)
openpose_model_hd.preprocessor.body_estimation.model.to('cuda')
parsing_model_hd = Parsing(0)
densepose_model_hd = DensePose4Gradio(
    cfg='preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    model='https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
)

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

# #### model init >>>>
config = OmegaConf.load("./configs/VITON.yaml")
config.model.params.img_H = IMG_H
config.model.params.img_W = IMG_W
params = config.model.params

# model = create_model(config_path=None, config=config)
# model.load_state_dict(torch.load("./checkpoints/eternal_1024.ckpt", map_location="cpu")["state_dict"])
# model = model.cuda()
# model.eval()
# sampler = PLMSSampler(model)

model2 = create_model(config_path=None, config=config)
model2.load_state_dict(torch.load("./checkpoints/VITONHD_1024.ckpt", map_location="cuda:0")["state_dict"])
model2 = model2.cuda()
model2.eval()
sampler2 = PLMSSampler(model2)
# #### model init <<<<

# @spaces.GPU
# @torch.autocast("cuda")
# @torch.no_grad()
# def stable_viton_model_hd(
#         batch,
#         n_steps,
# ):
#     z, cond = model.get_input(batch, params.first_stage_key)
#     z = z
#     bs = z.shape[0]
#     c_crossattn = cond["c_crossattn"][0][:bs]
#     if c_crossattn.ndim == 4:
#         c_crossattn = model.get_learned_conditioning(c_crossattn)
#         cond["c_crossattn"] = [c_crossattn]
#     uc_cross = model.get_unconditional_conditioning(bs)
#     uc_full = {"c_concat": cond["c_concat"], "c_crossattn": [uc_cross]}
#     uc_full["first_stage_cond"] = cond["first_stage_cond"]
#     for k, v in batch.items():
#         if isinstance(v, torch.Tensor):
#             batch[k] = v.cuda()
#     sampler.model.batch = batch

#     ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
#     start_code = model.q_sample(z, ts)
#     torch.cuda.empty_cache()
#     output, _, _ = sampler.sample(
#         n_steps,
#         bs,
#         (4, IMG_H//8, IMG_W//8),
#         cond,
#         x_T=start_code,
#         verbose=False,
#         eta=0.0,
#         unconditional_conditioning=uc_full,
#     )

#     output = model.decode_first_stage(output)
#     output = tensor2img(output)
#     pil_output = Image.fromarray(output)
#     return pil_output

ID = 1

@spaces.GPU
@torch.autocast("cuda")
@torch.no_grad()
def stable_viton_model_hd2(
        batch,
        n_steps,
):
    z, cond = model2.get_input(batch, params.first_stage_key)
    z = z
    bs = z.shape[0]
    c_crossattn = cond["c_crossattn"][0][:bs]
    if c_crossattn.ndim == 4:
        c_crossattn = model2.get_learned_conditioning(c_crossattn)
        cond["c_crossattn"] = [c_crossattn]
    uc_cross = model2.get_unconditional_conditioning(bs)
    uc_full = {"c_concat": cond["c_concat"], "c_crossattn": [uc_cross]}
    uc_full["first_stage_cond"] = cond["first_stage_cond"]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    sampler2.model.batch = batch

    ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
    start_code = model2.q_sample(z, ts)
    torch.cuda.empty_cache()
    output, _, _ = sampler2.sample(
        n_steps,
        bs,
        (4, IMG_H//8, IMG_W//8),
        cond,
        x_T=start_code,
        verbose=False,
        eta=0.0,
        unconditional_conditioning=uc_full,
    )

    output = model2.decode_first_stage(output)
    output = tensor2img(output)
    pil_output = Image.fromarray(output)
    return pil_output

@spaces.GPU
@torch.no_grad()
def process_hd(vton_img, garm_img, n_steps):
    global ID
    model_type = 'hd'
    category = 0  # 0:upperbody; 1:lowerbody; 2:dress

    stt = time.time()
    print('load images... ', end='')
    # garm_img = Image.open(garm_img).resize((IMG_W, IMG_H))
    # vton_img = Image.open(vton_img).resize((IMG_W, IMG_H))
    garm_img = Image.open(garm_img)
    vton_img = Image.open(vton_img)

    vton_img = center_crop(vton_img)
    garm_img = garm_img.resize((IMG_W, IMG_H))
    vton_img = vton_img.resize((IMG_W, IMG_H))

    print('%.2fs' % (time.time() - stt))

    stt = time.time()
    print('get agnostic map... ', end='')
    keypoints = openpose_model_hd(vton_img.resize((IMG_W, IMG_H)))
    model_parse, _ = parsing_model_hd(vton_img.resize((IMG_W, IMG_H)))
    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints, radius=5)
    mask = mask.resize((IMG_W, IMG_H), Image.NEAREST)
    mask_gray = mask_gray.resize((IMG_W, IMG_H), Image.NEAREST)
    masked_vton_img = Image.composite(mask_gray, vton_img, mask)  # agnostic map
    print('%.2fs' % (time.time() - stt))

    stt = time.time()
    print('get densepose... ', end='')
    vton_img = vton_img.resize((IMG_W, IMG_H))  # size for densepose
    densepose = densepose_model_hd.execute(vton_img)  # densepose
    print('%.2fs' % (time.time() - stt))

    batch = get_batch(
        vton_img,
        garm_img,
        densepose,
        masked_vton_img,
        mask,
        IMG_H,
        IMG_W
    )

    sample = stable_viton_model_hd2(
        batch,
        n_steps,
    )
    
    # Convert to white everything from sample that is outside of densepose
    densepose_mask = densepose.convert("L").point(lambda x: 255 if x > 0 else 0, mode='1')
    sample = Image.composite(sample, Image.new("RGB", sample.size, "white"), densepose_mask)

    sample.save(f"./stableviton-created_images/ID-{ID}.png", 'PNG')
    ID += 1

    return sample


example_path = opj(os.path.dirname(__file__), 'examples_eternal')
example_model_ps = sorted(glob(opj(example_path, "model/*")))
example_garment_ps = sorted(glob(opj(example_path, "garment/*")))

async def prepare_texture():
    global ID
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://smplitex:8000/{ID}") as response:
            if response.status == 200:
                ID += 1
                return await response.json()
            else:
                print(f"Error fetching images: {response.status}")
                return []
                

async def fetch_gallery_images(pose_id: int):
    """
    Asynchronous function to fetch image paths from the API.
    """
    global ID
    # call smplitex:8000/    httpx / requests
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://smplitex:8000/pose/{ID}/{pose_id}") as response:
            if response.status == 200:
                # Process the response to extract image paths
                return await response.json()  # Ensure you await the response.json() call
            else:
                print(f"Error fetching images: {response.status}")
                return []
            
async def get_image_from_3d_outputs(pose_id: int):
    """
    Asynchronous function to update the Gradio Gallery with image paths.
    """
    global ID
    # /3d_outputs
    output_images_path = sorted(glob(os.path.join(os.path.dirname(__file__), "3d_outputs/*")))
    target_file = next((file for file in output_images_path if f"ID-{ID-1}" in file and f"POSEID-{pose_id}" in file), None)
    img = Image.open(target_file)
    return img

async def load_gallery_images1():
    await load_gallery_images(1)
async def load_gallery_images2():
    await load_gallery_images(2)
async def load_gallery_images3():
    await load_gallery_images(3)

# New function to load images from output folder
async def load_gallery_images(pose_id: int):
    """
    Asynchronous task triggered by the button click.
    """
    print("Fetching images...")
    try:
        response = await fetch_gallery_images(pose_id)
        json_response = response.status
        print(json_response)
        print("Updating gallery...")
        image = get_image_from_3d_outputs(pose_id)
        
        return image
        # Return the list of image paths from the  output folder
        # output_images_path = sorted(glob(opj(os.path.dirname(__file__), "3d_outputs/*")))  # New path for output gallery images
    except aiohttp.ClientConnectionError as e:
        print(f"Connection error: {e}")
    except aiohttp.ClientError as e:
        print(f"Client error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # get images
    return

with gr.Blocks(css='style.css') as demo:
    with gr.Row():
        gr.Markdown("## Experience virtual try-on with your own images!")
    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="Model", type="filepath", height=384)
            example = gr.Examples(
                inputs=vton_img,
                examples_per_page=14,
                examples=example_model_ps)
        with gr.Column():
            garm_img = gr.Image(label="Garment", type="filepath", height=384)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=14,
                examples=example_garment_ps)
        with gr.Column():
            result_gallery_StableViton = gr.Image(label='Output', show_label=False, scale=1)
            # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)

    with gr.Column():
        run_button = gr.Button(value="Fit Garment")
        n_steps = gr.Slider(label="Steps", minimum=10, maximum=50, value=20, step=1)
        # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)

    ips = [vton_img, garm_img, n_steps]
    run_button.click(fn=process_hd, inputs=ips, outputs=[result_gallery_StableViton]).then(fn=prepare_texture)
    
    with gr.Row():
        posture1_button = gr.Button(value="Posture1")
        posture2_button = gr.Button(value="Posture2")
        posture3_button = gr.Button(value="Posture3")

    with gr.Row():
        with gr.Column():
             # Show output images from folder as a gallery
            result_gallery_SMPLitex = gr.Image(label='Output', show_label=False, scale=1)
    
    posture1_button.click(fn=load_gallery_images1, outputs = [result_gallery_SMPLitex])
    posture2_button.click(fn=load_gallery_images2, outputs = [result_gallery_SMPLitex])
    posture3_button.click(fn=load_gallery_images3, outputs = [result_gallery_SMPLitex])

    with gr.Row():
        gr.Markdown("Credit: StableVITON by rlawjdghek")


demo.queue().launch(share=True)