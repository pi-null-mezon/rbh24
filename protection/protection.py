import os
from easydict import EasyDict as edict
from tools import initialize_onnx_session, CustomDataset, extract_template_from_image, calculate_curves, \
    draw_rocs, convert_keypoints_to_bbox_normalized, encrypt_template, decrypt_template, \
    calculate_identifications, reconstruct_face_by_decoder, print_cosines_stats, extract_template_from_synth_image
from insightface.app import FaceAnalysis
from tqdm import tqdm
import numpy as np
import torch
import cv2

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda')

cfg = edict()
cfg.buffalo_cosine_threshold = 0.661  # measured for FMR 1E-6
cfg.visualize = True
cfg.visualize_ms = 30  # 0 - wait key press
cfg.faces_dataset = f"/home/{os.getlogin()}/Fastdata/HACK/geofaces"
cfg.max_ids = 1000
cfg.top_k = 10
cfg.decoder = '../train_naive_decoder/weights/buffalo_l_decoder_large_on_vgg11_v1.onnx'

buffalo_fa = FaceAnalysis(name='buffalo_l', root='../', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
buffalo_fa.prepare(ctx_id=0, det_size=(640, 640))

dataset = CustomDataset(photos_path=cfg.faces_dataset, max_ids=cfg.max_ids)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, drop_last=False, shuffle=False, num_workers=2)

print("TEMPLATES COLLECTION - please wait...")
registration_db = {}
identification_set_without_protection = []
for filenames, isreg, labels, photos in tqdm(dataloader):
    photos = photos.cpu().numpy()
    for i in range(len(filenames)):
        info = extract_template_from_image(photos[i], buffalo_fa)
        if info:
            biometric_data = {'emb': info['embedding'], 'kps': info['kps'], 'bbox': info['bbox']}
            if isreg[i]:
                biometric_data['filename'] = filenames[i]
                registration_db[labels[i]] = biometric_data
            else:
                biometric_data['id'] = labels[i]
                identification_set_without_protection.append(biometric_data)
print("TEMPLATES COLLECTION - finished")

print("DATABASE:")
print(f" - registration templates: {len(registration_db)}")
print(f" - identification templates: {len(identification_set_without_protection)}")


# let's calculate helper data - it is public (not secret) information that helps to decode encoded template
helper_data = []
for _id, item in registration_db.items():
    normalized_keypoints = convert_keypoints_to_bbox_normalized(keypoints=item['kps'], bbox=item['bbox'])
    helper_data.append(normalized_keypoints)
helper_data = np.array(helper_data).mean(axis=0)


protected_identification_set_encrypted = []
protected_identification_set_decrypted = []
cosines = []
for item in identification_set_without_protection:
    normalized_keypoints = convert_keypoints_to_bbox_normalized(keypoints=item['kps'], bbox=item['bbox'])
    encrypted_template = encrypt_template(item['emb'], normalized_keypoints)
    protected_identification_set_encrypted.append({'id': item['id'], 'emb': encrypted_template})
    decrypted_template = decrypt_template(encrypted_template, helper_data)
    protected_identification_set_decrypted.append({'id': item['id'], 'emb': decrypted_template})
    cosine = np.dot(encrypted_template / np.linalg.norm(encrypted_template), item['emb'] / np.linalg.norm(item['emb']))
    cosines.append(cosine)
print_cosines_stats(cosines, threshold=cfg.buffalo_cosine_threshold)


roc_without_protection = calculate_curves(matches=calculate_identifications(iset=identification_set_without_protection,
                                                                            regdb=registration_db,
                                                                            top_k=cfg.top_k))

roc_with_protection = calculate_curves(matches=calculate_identifications(iset=protected_identification_set_decrypted,
                                                                         regdb=registration_db,
                                                                         top_k=cfg.top_k))

draw_rocs([roc_without_protection, roc_with_protection],
          labels=['templates without protection', 'protected templates'],
          title=f'top {cfg.top_k} identifications for insightface/buffalo_l vs '
                f'{cfg.faces_dataset.rsplit("/", 1)[1]}:{len(registration_db)} ids '
                f'(rule of 3 applied)')

# Let's try to reconstruct encrypted vectors
decoder = initialize_onnx_session(cfg.decoder, use_cuda=True)
buffalo = initialize_onnx_session("../models/buffalo_l/w600k_r50.onnx", use_cuda=True)
cosines = []
for item in protected_identification_set_encrypted:
    rec = reconstruct_face_by_decoder(decoder, item['emb'])
    if cfg.visualize:
        regphoto_filename = os.path.join(cfg.faces_dataset, registration_db[item['id']]['filename'])
        reg = cv2.imread(regphoto_filename, cv2.IMREAD_COLOR)
        cv2.imshow("reconstructed", rec)
        cv2.imshow("registration_db", reg)
        cv2.waitKey(cfg.visualize_ms)
    nt_from_synth_face = extract_template_from_synth_image(rec, buffalo, normalize=True)
    reg_template = registration_db[item['id']]['emb']
    cosine = np.dot(nt_from_synth_face, reg_template / np.linalg.norm(reg_template))
    cosines.append(cosine)
print_cosines_stats(cosines, threshold=cfg.buffalo_cosine_threshold)
