import os
import base64
from easydict import EasyDict as edict
from tools import initialize_onnx_session, CustomDataSet, extract_template_from_image, load_image, \
    initialize_instantid_session, reconstruct_face_from_template, prepare_target_pose_kps, crop_square_roi, bbox_upscale
import numpy as np
import pickle
from argparse import ArgumentParser
from insightface.app import FaceAnalysis

argparser = ArgumentParser("validation script")
argparser.add_argument("--input", default="./input.jpg", help="source file to reconstruct (could be .jpg, .pkl or .b64)")
argparser.add_argument("--adapter", default="./models/buffalo2antelope_adapter_HQ_4000.onnx",
                       help="weights of adapter")
argparser.add_argument("--target_pose_photo", default="./examples/portrait1280p.jpg", help="photo to copy face pose")
argparser.add_argument("--output", default=f"./output.jpg", help="where to save generation result")
args = argparser.parse_args()

if not os.path.exists(args.input):
    print(f"Can not find input '{args.input}'! Abort...")
    exit()

cfg = edict()
cfg.target_reconstruction_size = (512, 512)
cfg.reconstruction_iterations = 40  # low values produce low quality results, high values took too long time

buffalo = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
buffalo.prepare(ctx_id=0, det_size=(640, 640))

ot = None
if '.jpg' in args.input or '.png' in args.input:
    info = extract_template_from_image(load_image(args.input), fa_model=buffalo)
    if len(info) == 0:
        print("Can not find face on input image! Abort....")
    ot = info['embedding']
elif '.pkl' in args.input:
    with open(args.input, 'rb') as i_f:
        ot = pickle.load(i_f)
elif '.b64' in args.input:
    with open(args.input, 'r') as i_f:
        bin_data = base64.b64decode(i_f.read().encode())
        ot = np.frombuffer(bin_data, dtype=np.float)

if ot is None:
    print("Template for face generation was not extracted or read from disk! Abort...")
    exit()

antelope = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
antelope.prepare(ctx_id=0, det_size=(640, 640))

adapter = initialize_onnx_session(model_path=args.adapter, use_cuda=True)

instantid = initialize_instantid_session()  # leave default arguments

target_pose_kps = prepare_target_pose_kps(args.target_pose_photo, antelope)

face_not_detected = True
while face_not_detected:
    pilimg = reconstruct_face_from_template(template=ot,
                                            instantid_session=instantid,
                                            adapter_session=adapter,
                                            target_kps=target_pose_kps,
                                            target_size=cfg.target_reconstruction_size,
                                            iterations=cfg.reconstruction_iterations)
    info = extract_template_from_image(pilimg, buffalo)
    if info is not None:
        rnt = info['embedding'] / np.linalg.norm(info['embedding'])
        face_not_detected = False

ont = ot / np.linalg.norm(ot)
cosine = np.dot(rnt, ont)
print(f"COSINE: {cosine:.4f}")
pilimg.save(args.output)
