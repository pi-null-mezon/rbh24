import os
import base64

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from argparse import ArgumentParser
from insightface.app import FaceAnalysis
from easydict import EasyDict as edict
from insightface.model_zoo.landmark import Landmark 

sys.path.insert(1, '../train_naive_decoder')
from tools import initialize_onnx_session, make_onnx_inference, extract_template_from_image, \
    extract_template_from_synth_image, tensor2image, norm_crop, arcface_dst


argparser = ArgumentParser("Tool for test encrypt generator on pipeline generation fake photo")
argparser.add_argument("--input", default="./input",
                       help="path to database with pair photo (files could be .jpg, .png)")
argparser.add_argument("--decoder", default="./weights/buffalo_l_decoder_large_on_vgg11_v1.onnx",
                       help="weights of decoder")
argparser.add_argument("--output", default=f"./result", help="where to save results of test encrypt")
args = argparser.parse_args()

if not os.path.exists(args.input):
    print(f"Can not find input '{args.input}'! Abort...")
    exit()
if not os.path.exists(args.output):
    os.makedirs(args.output)
    if not os.path.exists(args.output):
        print(f"Can not create output path '{args.output}'! Abort...")
        exit()



cfg = edict()
cfg.buffalo_cosine_threshold = 0.661  # measured for FMR 1E-6
cfg.mean = [0.485, 0.456, 0.406]  # torchvision default
cfg.std = [0.229, 0.224, 0.225]  # torchvision default
cfg.swap_red_blue = True  # torchvision default

buffalo_fa = FaceAnalysis(name='buffalo_l', root='../', providers=['CUDAExecutionProvider'])
buffalo_fa.prepare(ctx_id=0, det_size=(640, 640))

buffalo_lmk = Landmark(model_file='../models/buffalo_l/2d106det.onnx')

buffalo = initialize_onnx_session("../models/buffalo_l/w600k_r50.onnx", use_cuda=True)
decoder = initialize_onnx_session(args.decoder, use_cuda=True)

client_server = []
client_gen_server = []
client_encrypt_server = []
client_decrypt_server = []

for id in [i for i in os.listdir(args.input) if os.path.isdir(args.input + '/' + i)]:
    filename1, filename2 = os.listdir(args.input + '/' + id)

    ot_client = None
    ot_server = None
    abs_filename1 = os.path.join(args.input, id, filename1)
    abs_filename2 = os.path.join(args.input, id, filename2)
    if '.jpg' in filename1 or '.png' in filename1:
        source_image = cv2.imread(abs_filename1, cv2.IMREAD_COLOR)
        info = extract_template_from_image(source_image, fa_model=buffalo_fa)
        if info is None:
            print(f"Can not find face on input image! {abs_filename1}")
            continue
        ot_client = info['embedding']
        bbox = info['bbox'].astype(int)
        kps = info['kps']
        crop = source_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        kps[:, 0] = kps[:, 0] - bbox[0]
        kps[:, 1] = kps[:, 1] - bbox[1]
        align_crop = norm_crop(crop, kps)
        
        output = edict()
        output.bbox = [0, 0, 112, 112]
        kps_encrypt = buffalo_lmk.get(align_crop, output)[[37, 88, 86, 52, 61], :]
        kps_encrypt = np.tile(kps_encrypt.reshape(-1)/112, 52)[:512]
        ot_client_encrypt = ot_client - kps_encrypt
    else:
        continue
    
    if '.jpg' in filename2 or '.png' in filename2:
        source_image = cv2.imread(abs_filename2, cv2.IMREAD_COLOR)
        info = extract_template_from_image(source_image, fa_model=buffalo_fa)
        if info is None:
            print(f"Can not find face on input image! {abs_filename2}")
            continue
        ot_server = info['embedding']
    else:
        continue

    # format validation
    if ot_client is None:
        print(f"File: '{abs_filename1}' >> template was not extracted or can not be read from disk! Abort...")
        exit()
    if ot_server is None:
        print(f"File: '{abs_filename2}' >> template was not extracted or can not be read from disk! Abort...")
        exit()
    if not isinstance(ot_client, np.ndarray):
        print(f"File: '{abs_filename1}' >> template is not numpy array! Abort...")
        exit()
    if not isinstance(ot_server, np.ndarray):
        print(f"File: '{abs_filename2}' >> template is not numpy array! Abort...")
        exit()
    if ot_client.shape != (512,) and ot_client.shape != (1, 512):
        print(f"File: '{abs_filename1}' >> template have wrong shape {ot_client.shape}! Abort...")
        exit()
    if ot_server.shape != (512,) and ot_server.shape != (1, 512):
        print(f"File: '{abs_filename2}' >> template have wrong shape {ot_server.shape}! Abort...")
        exit()
    if 0.999 <= np.linalg.norm(ot_client) < 1.001:
        print(f"File: '{abs_filename1}' >> template should not be normalized ! Abort...")
        exit()
    if 0.999 <= np.linalg.norm(ot_server) < 1.001:
        print(f"File: '{abs_filename2}' >> template should not be normalized ! Abort...")
        exit()

    ot_server = np.reshape(ot_server, newshape=(512,))
    ont_server = ot_server / np.linalg.norm(ot_server)

    ot_client = np.reshape(ot_client, newshape=(1, 512))
    ont_client = ot_client / np.linalg.norm(ot_client)

    kps_decrypt = np.tile(arcface_dst.reshape(-1)/112, 52)[:512]
    ot_client_decrypt = ot_client_encrypt + kps_decrypt
    ot_client_decrypt = np.reshape(ot_client_decrypt, newshape=(1, 512))
    ont_client_decrypt = ot_client_decrypt / np.linalg.norm(ot_client_decrypt)

    ot_client_encrypt = np.reshape(ot_client_encrypt, newshape=(1, 512))
    ont_client_encrypt = ot_client_encrypt / np.linalg.norm(ot_client_encrypt)

    rec_client = make_onnx_inference(session=decoder, input_data=ont_client)
    rec_client = tensor2image(rec_client.squeeze(0), mean=cfg.mean, std=cfg.std, swap_red_blue=cfg.swap_red_blue)
    rnt_client = extract_template_from_synth_image(rec_client, buffalo_onnx_session=buffalo, normalize=True)

    rec_client_encrypt = make_onnx_inference(session=decoder, input_data=ont_client_encrypt)
    rec_client_encrypt = tensor2image(rec_client_encrypt.squeeze(0), mean=cfg.mean, std=cfg.std, swap_red_blue=cfg.swap_red_blue)
    rnt_client_encrypt = extract_template_from_synth_image(rec_client_encrypt, buffalo_onnx_session=buffalo, normalize=True)

    ont_client = np.reshape(ont_client, newshape=(512,))
    rnt_client = np.reshape(rnt_client, newshape=(512,))
    rnt_client_encrypt = np.reshape(rnt_client_encrypt, newshape=(512,))
    ont_client_decrypt = np.reshape(ont_client_decrypt, newshape=(512,))

    cosine1 = np.dot(ont_client, ont_server)
    cosine2 = np.dot(rnt_client, ont_server)
    cosine3 = np.dot(rnt_client_encrypt, ont_server)
    cosine4 = np.dot(ont_client_decrypt, ont_server)

    client_server.append(cosine1)
    client_gen_server.append(cosine2)
    client_encrypt_server.append(cosine3)
    client_decrypt_server.append(cosine4)


fig, ax = plt.subplots(figsize=(11,5))

data = {
    'Cosine': client_server + client_gen_server + client_encrypt_server + client_decrypt_server,
    'Type': 
    ['Оригинальные изображения' for _ in range(len(client_server))] +
    ['Реконструированные не защищённые шаблоны' for _ in range(len(client_gen_server))] +
    ['Реконструированные из защищённых шаблонов' for _ in range(len(client_encrypt_server))] +
    ['Защищенные шаблоны' for _ in range(len(client_decrypt_server))]
    ,
}

g = sns.kdeplot(
   data=data, x="Cosine", hue="Type",
   fill=True, common_norm=False,
   alpha=.5, linewidth=2,
)

plt.legend(title='Тип', loc='upper left', labels=['Оригинальные изображения', 'Реконструированные из защищённых шаблонов', 'Реконструированные не защищённые шаблоны', 'Защищенные шаблоны'])
ax.set_xlim(-1, 1)
plt.xticks([(i-10)*0.1 for i in range(21)] )
plt.grid(color = 'black', linewidth = 0.1)

abs_target_filename = os.path.join(args.output, f"test_encrypt.png")
plt.savefig(abs_target_filename)


