import os
from easydict import EasyDict as edict
from tools import image2tensor, tensor2image, initialize_onnx_session, make_onnx_inference, CustomDataSet
import numpy as np
import torch
import cv2

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')

cfg = edict()
cfg.visualize = False
cfg.visualize_ms = -1  # -1 - wait any key
cfg.mean = [0.485, 0.456, 0.406]  # torchvision default
cfg.std = [0.229, 0.224, 0.225]  # torchvision default
cfg.swap_red_blue = True  # torchvision default
cfg.crop_size = (128, 128)  # treat as (width, height)
cfg.normalize_templates = True
cfg.max_samples_to_collect = 1000

decoder = torch.load('./weights/_tmp_buffalo_decoder_on_vgg11.pth')
decoder.to(device)
decoder.eval()

cfg.buffalo_size = (112, 112)
cfg.buffalo_mean = 3 * [127.5 / 255]
cfg.buffalo_std = 3 * [127.5 / 255]
cfg.buffalo_swap_red_blue = True

buffalo = initialize_onnx_session('../models/buffalo_l/w600k_r50.onnx', use_cuda=False)


def extract_temlpate_from_synth_image(img, onnx_session, normalize):
    resized_img = cv2.resize(img, cfg.buffalo_size, interpolation=cv2.INTER_LINEAR)
    tensor = image2tensor(resized_img, cfg.buffalo_mean, cfg.buffalo_std, cfg.buffalo_swap_red_blue)
    tensor = np.expand_dims(tensor, axis=0)
    template = make_onnx_inference(onnx_session, tensor)
    if normalize:
        template = template / np.linalg.norm(template)
    return template


local_data_path = f"/home/{os.getlogin()}/Fastdata"
photos_path = f"{local_data_path}/OVISION/2T_2/datasets/recognition/train/glint360k/images"
test_templates_paths = [f"{local_data_path}/HACK/templatesgen_glint_0_10K.pkl"]

test_dataset = CustomDataSet(templates_paths=test_templates_paths,
                             photos_path=photos_path,
                             size=cfg.crop_size,
                             do_aug=False,
                             mean=cfg.mean,
                             std=cfg.std,
                             swap_reb_blue=cfg.swap_red_blue,
                             normalize_templates=cfg.normalize_templates)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)

print("MEASUREMENTS COLLECTION - please wait...")
cosines = []
with torch.no_grad():
    for template, photo in test_dataloader:

        template = template.squeeze(0)
        template = template.to(device)
        ont = template.cpu().numpy()
        probe = decoder(template)
        rec = tensor2image(probe.squeeze(0).cpu().numpy(),
                           mean=cfg.mean,
                           std=cfg.std,
                           swap_red_blue=cfg.swap_red_blue)
        snt = extract_temlpate_from_synth_image(rec, buffalo, normalize=True)
        cosine = np.dot(ont.squeeze(0), snt.squeeze(0)).item()
        cosines.append(cosine)
        if cfg.visualize:
            photo = photo.squeeze(0)
            orig = tensor2image(photo.cpu().numpy(),
                                mean=cfg.mean,
                                std=cfg.std,
                                swap_red_blue=cfg.swap_red_blue)

            canvas = np.zeros(shape=(orig.shape[0], orig.shape[1] + rec.shape[1], orig.shape[2]),
                              dtype=np.uint8)
            canvas[0:orig.shape[0], 0:orig.shape[1]] = orig
            canvas[0:rec.shape[0], orig.shape[1]:orig.shape[1] + rec.shape[1]] = rec
            info = f"cos: {cosine:.3f}"
            cv2.putText(canvas, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("probe", canvas)
            cv2.waitKey(cfg.visualize_ms)
        if len(cosines) >= cfg.max_samples_to_collect:
            break

print(f"STATISTICS ON {len(cosines)} TEST SAMPLES:")
cosines = np.array(cosines)
print(f" - COSINE MIN:    {cosines.min().item():.4f}")
print(f" - COSINE MEAN:   {cosines.mean().item():.4f}")
print(f" - COSINE MEDIAN: {np.median(cosines).item():.4f}")
print(f" - COSINE MAX:    {cosines.max().item():.4f}")
