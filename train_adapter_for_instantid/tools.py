from torch.utils.data import Dataset
import onnxruntime as ort
import pickle
import numpy as np
import cv2
import os
import torch
from PIL import Image
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


class CustomDataSet(Dataset):
    def __init__(self, templates_paths, photos_path, normalize_templates, max_samples_per_id=-1):
        assert max_samples_per_id != 0, "max_samples_per_id should not be equal to 0"
        self.photos_path = photos_path
        self.normalize_templates = normalize_templates
        self.templates = []
        self.filenames = []
        for filename in templates_paths:
            with open(filename, 'rb') as i_f:
                data = pickle.load(i_f)
                if max_samples_per_id == -1:
                    self.filenames += data['file']
                    self.templates += data['buffalo']
                else:
                    tmp = {}
                    for filename, template in zip(data['file'], data['buffalo']):
                        _id = filename.split('/', 1)[0]
                        if _id not in tmp:
                            tmp[_id] = [(filename, template)]
                        elif len(tmp[_id]) < max_samples_per_id:
                            tmp[_id].append((filename, template))
                    for _id in tmp:
                        for filename, template in tmp[_id]:
                            self.filenames.append(filename)
                            self.templates.append(template)

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, idx):
        filename = os.path.join(self.photos_path, self.filenames[idx])
        mat = cv2.imread(filename, cv2.IMREAD_COLOR)
        template = self.templates[idx]
        if self.normalize_templates:
            template = template / np.linalg.norm(template)
        return template, mat


def initialize_onnx_session(model_path, use_cuda=True):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    try:
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        if use_cuda:
            assert 'CUDAExecutionProvider' in session.get_providers(), "CUDA is not available"
        print(f"ONNX Runtime session initialized with providers: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"Error initializing ONNX Runtime session: {str(e)}")
        return None


def make_onnx_inference(session, input_data):
    results = session.run(None, {session.get_inputs()[0].name: input_data})
    return results[0]


def resize_img(input_image, max_side=1280, min_side=1280, size=None, pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)
    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def initialize_instantid_session(diffusion_model='wangqixun/YamerMIX_v8',
                                 face_adapter='./checkpoints/ip-adapter.bin',
                                 controlnet='./checkpoints/ControlNetModel'):
    assert torch.cuda.is_available(), "InstantID can not be run without CUDA"
    controlnet = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        diffusion_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)
    return pipe


def prepare_target_pose_kps(filename, fa_model):
    pose_image = resize_img(load_image(filename))
    info = fa_model.get(cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR))
    info = sorted(info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # biggest one
    return draw_kps(pose_image, info['kps'])


def reconstruct_face_from_template(template, instantid_session, adapter_session, target_kps, target_size,
                                   iterations=40):
    if template.shape == (512,):
        template = template.reshape(1, 512)
    antelope_face_emb = make_onnx_inference(session=adapter_session, input_data=template)
    antelope_face_emb = antelope_face_emb.reshape((512,))
    image = instantid_session(
        prompt="regular portrait, professional, 4k, highly detailed, white background, hyper-realistic",
        negative_prompt="drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, "
                        "deformed, ugly, text, words, naked, hands, occlusion, sketch, muscules, neon, glow, "
                        "overexposed",
        image_embeds=antelope_face_emb,
        image=target_kps,
        controlnet_conditioning_scale=0.9,
        ip_adapter_scale=0.9,
        num_inference_steps=iterations,
        guidance_scale=5,
    ).images[0]
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    return image


def extract_template_from_image(image, fa_model):
    info = fa_model.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    if len(info) > 0:
        info = sorted(info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # biggest one
        return info
    return None


def bbox_upscale(image, bbox, resize_factor):
    # Unpack the bounding box coordinates
    x_tl, y_tl, x_br, y_br = bbox

    # Calculate the original width and height of the bounding box
    width = x_br - x_tl
    height = y_br - y_tl

    # Calculate the center of the bounding box
    center_x = (x_tl + x_br) // 2
    center_y = (y_tl + y_br) // 2

    # Compute the new width and height after scaling
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)

    # Calculate the new top-left and bottom-right coordinates
    new_x_tl = center_x - new_width // 2
    new_y_tl = center_y - new_height // 2
    new_x_br = center_x + new_width // 2
    new_y_br = center_y + new_height // 2

    # Ensure the new bounding box coordinates are within the image bounds
    new_x_tl = max(0, new_x_tl)
    new_y_tl = max(0, new_y_tl)
    new_x_br = min(image.shape[1], new_x_br)  # shape[1] is the image width
    new_y_br = min(image.shape[0], new_y_br)  # shape[0] is the image height

    return [new_x_tl, new_y_tl, new_x_br, new_y_br]


def crop_square_roi(image, bbox, v2hshift):
    # Unpack the bounding box coordinates
    x_tl, y_tl, x_br, y_br = bbox

    y_shift = int(v2hshift * (y_br-y_tl))
    y_tl += y_shift
    y_br += y_shift

    # Calculate the width and height of the bounding box
    width = x_br - x_tl
    height = y_br - y_tl

    # Determine the size of the square (max of width and height)
    square_size = max(width, height)

    # Calculate the center of the bounding box
    center_x = (x_tl + x_br) // 2
    center_y = (y_tl + y_br) // 2

    # Calculate the new top-left and bottom-right coordinates for the square
    new_x_tl = center_x - square_size // 2
    new_y_tl = center_y - square_size // 2
    new_x_br = center_x + square_size // 2
    new_y_br = center_y + square_size // 2

    # Ensure the square coordinates are within image bounds
    new_x_tl = int(max(0, new_x_tl))
    new_y_tl = int(max(0, new_y_tl))
    new_x_br = int(min(image.shape[1], new_x_br))  # shape[1] is the width of the image
    new_y_br = int(min(image.shape[0], new_y_br))  # shape[0] is the height of the image

    # Crop the image using the new square coordinates
    cropped_image = image[new_y_tl:new_y_br, new_x_tl:new_x_br]

    return cropped_image
