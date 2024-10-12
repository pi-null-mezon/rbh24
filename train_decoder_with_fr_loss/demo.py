import os
import base64

import cv2
import numpy as np
import pickle
from tools import initialize_onnx_session, make_onnx_inference, extract_template_from_image, \
    extract_template_from_synth_image, tensor2image
from argparse import ArgumentParser
from insightface.app import FaceAnalysis
from easydict import EasyDict as edict


argparser = ArgumentParser("Tool to convert buffalo_l templates (non normalized) into photo of the face")
argparser.add_argument("--input", default="../examples",
                       help="path to files to reconstruct (files could be .jpg, .png, .pkl or .b64)")
argparser.add_argument("--decoder", default="./weights/buffalo_decoder_on_fr_wo_discr_wo_pixel_loss_last.onnx",
                       help="weights of decoder")
argparser.add_argument("--output", default=f"../examples/decoder", help="where to save generation results")
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
cfg.mean = [0.5, 0.5, 0.5]
cfg.std = [0.5, 0.5, 0.5]
cfg.swap_red_blue = True  # torchvision default

buffalo_fa = FaceAnalysis(name='buffalo_l', root='../', providers=['CPUExecutionProvider'])
buffalo_fa.prepare(ctx_id=0, det_size=(640, 640))

buffalo = initialize_onnx_session("../models/buffalo_l/w600k_r50.onnx", use_cuda=False)
decoder = initialize_onnx_session(args.decoder, use_cuda=False)

cosines = []
for filename in [f.name for f in os.scandir(args.input) if f.is_file()]:
    ot = None
    abs_filename = os.path.join(args.input, filename)
    if '.jpg' in filename or '.png' in filename:
        source_image = cv2.imread(abs_filename, cv2.IMREAD_COLOR)
        info = extract_template_from_image(source_image, fa_model=buffalo_fa)
        if info is None:
            print(f"Can not detect face on '{abs_filename}'! File will be skipped...")
            continue
        ot = info['embedding']
    elif '.pkl' in filename:
        with open(args.input, 'rb') as i_f:
            ot = pickle.load(i_f)
    elif '.b64' in filename:
        with open(args.input, 'r') as i_f:
            bin_data = base64.b64decode(i_f.read().encode())
            ot = np.frombuffer(bin_data, dtype=np.float)
    else:
        continue
    # format validation
    if ot is None:
        print(f"File: '{abs_filename}' >> template was not extracted or can not be read from disk! Abort...")
        exit()
    if not isinstance(ot, np.ndarray):
        print(f"File: '{abs_filename}' >> template is not numpy array! Abort...")
        exit()
    if ot.shape != (512,) and ot.shape != (1, 512):
        print(f"File: '{abs_filename}' >> template have wrong shape {ot.shape}! Abort...")
        exit()
    if 0.999 <= np.linalg.norm(ot) < 1.001:
        print(f"File: '{abs_filename}' >> template should not be normalized ! Abort...")
        exit()

    ot = np.reshape(ot, newshape=(1, 512))
    ont = ot / np.linalg.norm(ot)
    rec = make_onnx_inference(session=decoder, input_data=ont)
    rec = tensor2image(rec.squeeze(0), mean=cfg.mean, std=cfg.std, swap_red_blue=cfg.swap_red_blue)
    rnt = extract_template_from_synth_image(rec, buffalo_onnx_session=buffalo, normalize=True)
    rnt = np.reshape(rnt, newshape=(512,))
    ont = np.reshape(ont, newshape=(512,))
    cosine = np.dot(rnt, ont)
    cosines.append(cosine)
    print(f"For {abs_filename} we have got COSINE: {cosine:.4f}")
    abs_target_filename = os.path.join(args.output, filename.rsplit('.', 1)[0] + f"_(cosine {cosine:.4f}).png")
    cv2.imwrite(abs_target_filename, rec)

if len(cosines) > 0:
    print(f"STATISTICS ON {len(cosines)} SAMPLES FROM '{args.input}':")
    cosines = np.array(cosines)
    print(f" - COSINE MIN:    {cosines.min().item():.4f}")
    print(f" - COSINE MEAN:   {cosines.mean().item():.4f}")
    print(f" - COSINE MEDIAN: {np.median(cosines).item():.4f}")
    print(f" - COSINE MAX:    {cosines.max().item():.4f}")
    tp = np.sum(cosines > cfg.buffalo_cosine_threshold)
    print(f"TOTAL: {tp} of {len(cosines)} have cosine with genuine template greater than {cfg.buffalo_cosine_threshold:.3f}"
          f" >> it is {100*tp/len(cosines):.1f} % of enrolled samples")