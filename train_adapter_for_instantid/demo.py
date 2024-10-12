import os
import base64
from easydict import EasyDict as edict
from tools import initialize_onnx_session, extract_template_from_image, load_image, setup_logger, \
    initialize_instantid_session, reconstruct_face_from_template, prepare_target_pose_kps
import numpy as np
import pickle
from argparse import ArgumentParser
from insightface.app import FaceAnalysis

argparser = ArgumentParser("Tool to convert buffalo_l templates (non normalized) into photo of the face")
argparser.add_argument("--input", default="./input",
                       help="path to files to reconstruct (files could be .jpg, .png, .pkl or .b64)")
argparser.add_argument("--adapter", default="./models/buffalo2atelope_adapter_analytical.onnx",
                       help="weights of adapter")
argparser.add_argument("--target_pose_photo", default="./examples/portrait.jpg", help="photo to copy face pose")
argparser.add_argument("--reconstruction_iterations", type=int, default=15, help="diffusion iterations to do")
argparser.add_argument("--add_cos_in_filename", action="store_true", help="add cosine value to output filename")
argparser.add_argument("--output", default=f"./output", help="where to save generation results")
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

logger = setup_logger(os.path.join(args.output, 'reconstruction.log'), None)

buffalo = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
buffalo.prepare(ctx_id=0, det_size=(640, 640))

antelope = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
antelope.prepare(ctx_id=0, det_size=(640, 640))

adapter = initialize_onnx_session(model_path=args.adapter, use_cuda=True)

instantid = initialize_instantid_session()  # leave default arguments

target_pose_kps = prepare_target_pose_kps(args.target_pose_photo, antelope)

read_templates_from_pickle = False
if '.pickle' in args.input:
    with open(args.input, 'rb') as i_f:
        pickle_templates = pickle.load(i_f)
        read_templates_from_pickle = True
if read_templates_from_pickle:
    filenames_list = pickle_templates.keys()
else:
    filenames_list = [f.name for f in os.scandir(args.input) if f.is_file()]

cosines = []
for filename in filenames_list:
    ot = None
    if not read_templates_from_pickle:
        abs_filename = os.path.join(args.input, filename)
        if '.jpg' in abs_filename or '.jpeg' in abs_filename or '.png' in abs_filename:
            info = extract_template_from_image(load_image(abs_filename), fa_model=buffalo)
            if info is None:
                logger.info(f"Can not detect face on '{abs_filename}'! File will be skipped...")
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
    else:
        ot = pickle_templates[filename]['embedding']
    # format validation
    if ot is None:
        logger.info(f"File: '{abs_filename}' >> template was not extracted or can not be read from disk! Abort...")
        exit()
    if not isinstance(ot, np.ndarray):
        logger.info(f"File: '{abs_filename}' >> template is not numpy array! Abort...")
        exit()
    if ot.shape != (512,) and ot.shape != (1, 512):
        logger.info(f"File: '{abs_filename}' >> template have wrong shape {ot.shape}! Abort...")
        exit()
    if 0.999 <= np.linalg.norm(ot) < 1.001:
        logger.info(f"File: '{abs_filename}' >> template should not be normalized ! Abort...")
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

    if face_not_detected:
        cosines.append(0)
        logger.info(f"For {abs_filename} we could not reconstruct face within max allowed attempts :(")
    else:
        ont = ot / np.linalg.norm(ot)
        cosine = np.dot(rnt, ont)
        cosines.append(cosine)
        logger.info(f"For {filename} we have got COSINE: {cosine:.4f}")
        if args.add_cos_in_filename:
            abs_target_filename = os.path.join(args.output, filename.rsplit('.', 1)[0] + f"_(cosine {cosine:.4f}).jpg")
        else:
            abs_target_filename = os.path.join(args.output, filename)
        pilimg.save(abs_target_filename)


if len(cosines) > 0:
    logger.info(f"STATISTICS ON {len(cosines)} TEST SAMPLES FROM '{args.input}':")
    cosines = np.array(cosines)
    logger.info(f" - COSINE MIN:    {cosines.min().item():.4f}")
    logger.info(f" - COSINE MEAN:   {cosines.mean().item():.4f}")
    logger.info(f" - COSINE MEDIAN: {np.median(cosines).item():.4f}")
    logger.info(f" - COSINE MAX:    {cosines.max().item():.4f}")
    tp = np.sum(cosines > cfg.buffalo_cosine_threshold)
    logger.info(f"TOTAL: {tp} of {len(cosines)} have cosine with genuine template greater than {cfg.buffalo_cosine_threshold:.3f}"
                f" >> it is {100*tp/len(cosines):.1f} % of enrolled samples")
    logger.info(f"\n-------------------------------------------------\nSUM OF ALL COSINES: {cosines.sum().item():.4f}")
