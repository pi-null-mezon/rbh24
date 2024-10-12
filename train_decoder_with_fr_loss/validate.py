import os
from easydict import EasyDict as edict
from tools import tensor2image, initialize_onnx_session, CustomDataSet, extract_template_from_synth_image
from tqdm import tqdm
import numpy as np
import torch
import cv2
from argparse import ArgumentParser


argparser = ArgumentParser("validation script")
argparser.add_argument("--set", choices={'glint', 'valface'}, help="name of set to validate on")
argparser.add_argument("--max_ids", type=int, default=-1, help="max ids to process (-1 - take all)")
args = argparser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda')

cfg = edict()
cfg.buffalo_cosine_threshold = 0.661  # measured for FMR 1E-6
cfg.visualize = True
cfg.visualize_ms = 1  # 0 - wait key press
cfg.size_for_visualization = (256, 256)
cfg.mean = [0.5, 0.5, 0.5]
cfg.std = [0.5, 0.5, 0.5]
cfg.swap_red_blue = True  # torchvision default
cfg.crop_size = (128, 128)  # treat as (width, height)
cfg.normalize_templates = True
cfg.max_samples_to_collect = args.max_ids  # we will take 1 sample per id

decoder = torch.load('./weights/_tmp_buffalo_decoder_on_fr_wo_discr_wo_pixel_loss_last.pth')
decoder.to(device)
decoder.eval()

buffalo = initialize_onnx_session('../models/buffalo_l/w600k_r50.onnx', use_cuda=torch.cuda.is_available())

local_data_path = f"/home/{os.getlogin()}/Fastdata"
if args.set == 'glint':
    photos_path = f"{local_data_path}/OVISION/2T_2/datasets/recognition/train/glint360k/images"
    test_templates_paths = [f"{local_data_path}/HACK/templatesgen_glint_0_10K.pkl"]
else:
    photos_path = f"{local_data_path}/HACK/valface/crop"
    test_templates_paths = [f"{local_data_path}/HACK/valface/templatesgen_valface.pkl"]

test_dataset = CustomDataSet(templates_paths=test_templates_paths,
                             photos_path=photos_path,
                             size=cfg.crop_size,
                             do_aug=False,
                             mean=cfg.mean,
                             std=cfg.std,
                             swap_reb_blue=cfg.swap_red_blue,
                             normalize_templates=cfg.normalize_templates,
                             max_samples_per_id=1)

cfg.max_samples_to_collect = min(cfg.max_samples_to_collect, len(test_dataset))
if cfg.max_samples_to_collect == -1:
    cfg.max_samples_to_collect = len(test_dataset)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

print("MEASUREMENTS COLLECTION - please wait...", flush=True)
cosines = []
with torch.no_grad(), tqdm(total=cfg.max_samples_to_collect) as pbar:
    for template, photo in test_dataloader:

        template = template.squeeze(0)
        template = template.to(device)
        ont = template.cpu().numpy()
        probe = decoder(template)
        rec = tensor2image(probe.squeeze(0).cpu().numpy(),
                           mean=cfg.mean,
                           std=cfg.std,
                           swap_red_blue=cfg.swap_red_blue)
        snt = extract_template_from_synth_image(rec, buffalo, normalize=True)
        cosine = np.dot(ont.squeeze(0), snt.squeeze(0)).item()
        cosines.append(cosine)
        if cfg.visualize:
            photo = photo.squeeze(0)
            orig = tensor2image(photo.cpu().numpy(),
                                mean=cfg.mean,
                                std=cfg.std,
                                swap_red_blue=cfg.swap_red_blue)
            orig = cv2.resize(orig, cfg.size_for_visualization, cv2.INTER_CUBIC)
            rec = cv2.resize(rec, cfg.size_for_visualization, cv2.INTER_CUBIC)
            canvas = np.zeros(shape=(orig.shape[0], orig.shape[1] + rec.shape[1], orig.shape[2]),
                              dtype=np.uint8)
            canvas[0:orig.shape[0], 0:orig.shape[1]] = orig
            canvas[0:rec.shape[0], orig.shape[1]:orig.shape[1] + rec.shape[1]] = rec
            info = f"cosine: {cosine:.3f}"
            color = (0, 255, 0) if cosine > cfg.buffalo_cosine_threshold else (0, 55, 255)
            cv2.putText(canvas, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, info, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.imshow("probe", canvas)
            cv2.waitKey(cfg.visualize_ms)
        pbar.update(1)
        if len(cosines) >= cfg.max_samples_to_collect:
            break

print(f"STATISTICS ON {len(cosines)} TEST SAMPLES FROM '{args.set}':")
cosines = np.array(cosines)
print(f" - COSINE MIN:    {cosines.min().item():.4f}")
print(f" - COSINE MEAN:   {cosines.mean().item():.4f}")
print(f" - COSINE MEDIAN: {np.median(cosines).item():.4f}")
print(f" - COSINE MAX:    {cosines.max().item():.4f}")
tp = np.sum(cosines > cfg.buffalo_cosine_threshold)
print(f"TOTAL: {tp} of {len(cosines)} have cosine with genuine template greater than {cfg.buffalo_cosine_threshold:.3f}"
      f" >> it is {100*tp/len(cosines):.1f} % of validation samples")
