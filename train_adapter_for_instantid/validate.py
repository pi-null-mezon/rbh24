import os
import cv2
from easydict import EasyDict as edict
from tools import initialize_onnx_session, CustomDataSet, extract_template_from_image, \
    initialize_instantid_session, reconstruct_face_from_template, prepare_target_pose_kps, crop_square_roi, bbox_upscale
import numpy as np
import torch
import pickle
from argparse import ArgumentParser
from insightface.app import FaceAnalysis

argparser = ArgumentParser("validation script")
argparser.add_argument("--set", choices={'glint', 'valface'}, help="name of set to validate on")
argparser.add_argument("--max_ids", type=int, default=-1, help="max ids to process (-1 - take all)")
argparser.add_argument("--adapter", default="./models/buffalo2atelope_adapter_analytical.onnx",
                       help="weights of adapter")
argparser.add_argument("--target_pose_photo", default="./examples/portrait1280p.jpg", help="photo to copy face pose")
args = argparser.parse_args()

adapter_name = args.adapter.rsplit('.onnx', 1)[0].rsplit('/', 1)[1]
artifacts_path = os.path.join('./validation/', f"{adapter_name}_vs_{args.set}")
if os.path.exists(artifacts_path):
    print(f"It seems this adapter already validated. Delete artifacts in '{artifacts_path}' if you want to revalidate")
    exit()
else:
    os.makedirs(artifacts_path)

cfg = edict()
cfg.buffalo_cosine_threshold = 0.661  # measured for FMR 1E-6
cfg.visualize = True
cfg.visualize_ms = 30  # 0 - wait key press
cfg.size_for_visualization = (256, 256)
cfg.target_reconstruction_size = (512, 512)
cfg.reconstruction_iterations = 11  # low values produce low quality results, high values took too long time
cfg.max_samples_to_collect = args.max_ids  # we will take 1 sample per id
cfg.max_allowed_attempts_to_generate_face = 4  # sometime diffusion could not generate face (wrong template projection)

buffalo = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
buffalo.prepare(ctx_id=0, det_size=(640, 640))

antelope = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
antelope.prepare(ctx_id=0, det_size=(640, 640))

adapter = initialize_onnx_session(model_path=args.adapter, use_cuda=True)

instantid = initialize_instantid_session()  # leave default arguments

target_pose_kps = prepare_target_pose_kps(args.target_pose_photo, antelope)

local_data_path = f"/home/{os.getlogin()}/Fastdata"
if args.set == 'glint':
    photos_path = f"{local_data_path}/OVISION/2T_2/datasets/recognition/train/glint360k/images"
    test_templates_paths = [f"{local_data_path}/HACK/templatesgen_glint_0_10K.pkl"]
else:
    photos_path = f"{local_data_path}/HACK/valface/crop"
    test_templates_paths = [f"{local_data_path}/HACK/valface/templatesgen_valface.pkl"]

test_dataset = CustomDataSet(templates_paths=test_templates_paths,
                             photos_path=photos_path,
                             normalize_templates=False,
                             max_samples_per_id=1)

cfg.max_samples_to_collect = min(cfg.max_samples_to_collect, len(test_dataset))
if cfg.max_samples_to_collect == -1:
    cfg.max_samples_to_collect = len(test_dataset)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

print("MEASUREMENTS COLLECTION - please wait...", flush=True)
cosines = []
for template, photo, filename in test_dataloader:
    ot = template.squeeze(0).cpu().numpy()
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
        ont = ont.squeeze(0)
        cosine = np.dot(rnt, ont)
        cosines.append(cosine)

        # save generation locally
        _id = filename[0].split('/', 1)[0]
        os.makedirs(os.path.join(artifacts_path, _id))
        pilimg.save(f'{os.path.join(artifacts_path, filename[0].rsplit(".",1)[0])}.jpg')

        if cfg.visualize:
            orig = photo.squeeze(0).cpu().numpy()
            orig = cv2.resize(orig, cfg.size_for_visualization, cv2.INTER_CUBIC)
            rec = np.array(pilimg)
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            rec = crop_square_roi(rec, bbox_upscale(image=rec, bbox=info['bbox'], resize_factor=1.1), v2hshift=-0.075)
            rec = cv2.resize(rec, cfg.size_for_visualization, cv2.INTER_CUBIC)
            canvas = np.zeros(shape=(orig.shape[0], orig.shape[1] + rec.shape[1], orig.shape[2]), dtype=np.uint8)
            canvas[0:orig.shape[0], 0:orig.shape[1]] = orig
            canvas[0:rec.shape[0], orig.shape[1]:orig.shape[1] + rec.shape[1]] = rec
            info = f"cosine: {cosine:.3f}"
            color = (0, 255, 0) if cosine > cfg.buffalo_cosine_threshold else (0, 55, 255)
            cv2.putText(canvas, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, info, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.imshow("probe", canvas)
            cv2.waitKey(cfg.visualize_ms)
        if len(cosines) >= cfg.max_samples_to_collect:
            break
    else:
        cosines.append(0)


cosines = np.array(cosines)
with open(os.path.join(artifacts_path, f'{adapter_name}.cosines.pkl'), 'wb') as o_f:
    pickle.dump(cosines, o_f)
line = f"STATISTICS ON {len(cosines)} TEST SAMPLES FROM '{args.set}':" \
       f"\n - COSINE MIN:    {cosines.min().item():.4f}" \
       f"\n - COSINE MEAN:   {cosines.mean().item():.4f}" \
       f"\n - COSINE MEDIAN: {np.median(cosines).item():.4f}" \
       f"\n - COSINE MAX:    {cosines.max().item():.4f}"
tp = np.sum(cosines > cfg.buffalo_cosine_threshold)
line += f"\nTOTAL: {tp} of {len(cosines)} have cosine with genuine template greater than {cfg.buffalo_cosine_threshold:.3f}" \
        f" >> it is {100 * tp / len(cosines):.1f} % of validation samples"
with open(os.path.join(artifacts_path, f'{adapter_name}.validation.md'), 'w') as o_f:
    o_f.write(line)
print(line)
