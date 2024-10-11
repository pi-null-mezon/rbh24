import os
import base64
from easydict import EasyDict as edict
from tools import initialize_onnx_session, extract_template_from_image, load_image, \
    initialize_instantid_session, reconstruct_face_from_template, prepare_target_pose_kps
import numpy as np
import pickle
from argparse import ArgumentParser
from insightface.app import FaceAnalysis

argparser = ArgumentParser("Tool to convert buffalo_l templates (non normalized) into photo of the face")
argparser.add_argument("--input", default="./examples",
                       help="path to files to reconstruct (files could be .jpg, .png, .pkl or .b64)")
argparser.add_argument("--adapter", default="./models/buffalo2antelope_adapter_100K.onnx",
                       help="weights of adapter")
argparser.add_argument("--target_pose_photo", default="./examples/portrait1280p.jpg", help="photo to copy face pose")
argparser.add_argument("--reconstruction_iterations", type=int, default=16, help="diffusion iterations to make")
argparser.add_argument("--output", default=f"./examples/adapter", help="where to save generation results")
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
cfg.max_allowed_attempts_to_generate_face = 4  # sometime diffusion could not generate face (wrong template projection)
cfg.target_reconstruction_size = (512, 512)
cfg.reconstruction_iterations = args.reconstruction_iterations

buffalo = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
buffalo.prepare(ctx_id=0, det_size=(640, 640))

antelope = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
antelope.prepare(ctx_id=0, det_size=(640, 640))

adapter = initialize_onnx_session(model_path=args.adapter, use_cuda=True)

instantid = initialize_instantid_session()  # leave default arguments

target_pose_kps = prepare_target_pose_kps(args.target_pose_photo, antelope)

cosines = []
for filename in [f.name for f in os.scandir(args.input) if f.is_file()]:
    ot = None
    abs_filename = os.path.join(args.input, filename)
    if '.jpg' in abs_filename or '.jpeg' in abs_filename or '.png' in abs_filename:
        info = extract_template_from_image(load_image(abs_filename), fa_model=buffalo)
        if info is None:
            print(f"Can not detect face on '{abs_filename}'! File will be skipped...")
            continue
        ot = info['embedding']
    elif '.pkl' in abs_filename:
        with open(abs_filename, 'rb') as i_f:
            ot = pickle.load(i_f)
    elif '.b64' in abs_filename:
        with open(abs_filename, 'r') as i_f:
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

    fail_iterations = 0
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
        fail_iterations += 1
        if fail_iterations > cfg.max_allowed_attempts_to_generate_face:
            break
    if not face_not_detected:
        ont = ot / np.linalg.norm(ot)
        cosine = np.dot(rnt, ont)
        cosines.append(cosine)
        print(f"For {abs_filename} we have got COSINE: {cosine:.4f}")
        abs_target_filename = os.path.join(args.output, filename.rsplit('.', 1)[0] + f"_(cosine {cosine:.4f}).jpg")
        pilimg.save(abs_target_filename)
    else:
        cosines.append(0)
        print(f"For {abs_filename} we could not reconstruct face within max allowed attempts :(")

if len(cosines) > 0:
    print(f"STATISTICS ON {len(cosines)} TEST SAMPLES FROM '{args.input}':")
    cosines = np.array(cosines)
    print(f" - COSINE MIN:    {cosines.min().item():.4f}")
    print(f" - COSINE MEAN:   {cosines.mean().item():.4f}")
    print(f" - COSINE MEDIAN: {np.median(cosines).item():.4f}")
    print(f" - COSINE MAX:    {cosines.max().item():.4f}")
    tp = np.sum(cosines > cfg.buffalo_cosine_threshold)
    print(f"TOTAL: {tp} of {len(cosines)} have cosine with genuine template greater than {cfg.buffalo_cosine_threshold:.3f}"
          f" >> it is {100*tp/len(cosines):.1f} % of enrolled samples")
