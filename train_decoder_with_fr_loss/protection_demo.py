import os
from easydict import EasyDict as edict
from tools import initialize_onnx_session, extract_template_from_synth_image, extract_template_from_image,\
    fit_img_into_rectangle
from protection import CustomDataset, calculate_curves, \
    draw_rocs, convert_keypoints_to_bbox_normalized, encrypt_template, decrypt_template, \
    calculate_identifications, reconstruct_face_by_decoder, print_cosines_stats
from insightface.app import FaceAnalysis
from tqdm import tqdm
import numpy as np
import torch
import cv2

# -------- CONFIG ---------

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda')

cfg = edict()
cfg.buffalo_cosine_threshold = 0.661  # measured for FMR 1E-6
cfg.visualize = True
cfg.visualize_ms = 30  # 0 - wait key press
cfg.faces_dataset = f"/home/{os.getlogin()}/Fastdata/HACK/geofaces_plus_opentest"
cfg.max_ids = -1  # -1 means collect all
cfg.top_k = 10
cfg.decoder = './weights/buffalo_decoder_on_fr_wo_discr_wo_pixel_loss_last.onnx'

buffalo_fa = FaceAnalysis(name='buffalo_l', root='../', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
buffalo_fa.prepare(ctx_id=0, det_size=(640, 640))

dataset = CustomDataset(photos_path=cfg.faces_dataset, max_ids=cfg.max_ids)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=2)

# -------- LOGIC -----------

# Step 1 - enroll local database of photos

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
print("TEMPLATES COLLECTION - finished\n")
print("DATABASE:")
print(f" - registration templates: {len(registration_db)}")
print(f" - identification templates: {len(identification_set_without_protection)}\n")

# Step 2 - calculate helper_data on database registration photos collection
# helper data is public (not secret) information that helps to decode encoded template - it is stored in database
# In our solution we will use facial landmarks coordinates - they are random enough to be key for our encryption

helper_data = []
for _id, item in registration_db.items():
    normalized_keypoints = convert_keypoints_to_bbox_normalized(keypoints=item['kps'], bbox=item['bbox'])
    helper_data.append(normalized_keypoints)
helper_data = np.array(helper_data).mean(axis=0)

# Step 3 - calculate encrypted identification templates and decrypted identification templates to simulate transfer
# over open channel (for the instance internet)

protected_identification_set_encrypted = []
protected_identification_set_decrypted = []
cosines = []
print("ENCRYPTING TEMPLATES - please wait...")
for item in identification_set_without_protection:
    normalized_keypoints = convert_keypoints_to_bbox_normalized(keypoints=item['kps'], bbox=item['bbox'])
    encrypted_template = encrypt_template(item['emb'], normalized_keypoints)
    protected_identification_set_encrypted.append({'id': item['id'], 'emb': encrypted_template})
    decrypted_template = decrypt_template(encrypted_template, helper_data)
    protected_identification_set_decrypted.append({'id': item['id'], 'emb': decrypted_template})
    cosine = np.dot(encrypted_template / np.linalg.norm(encrypted_template), item['emb'] / np.linalg.norm(item['emb']))
    cosines.append(cosine)
print_cosines_stats(cosines, threshold=cfg.buffalo_cosine_threshold)
print("ENCRYPTING TEMPLATES - finished\n")

# Step 4 - let's test our identification system ROC characteristic (FPIR vs FNIR) with and without encryption

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

# Step 5 - Let's try to reconstruct encrypted templates

print("CHECK POSITIVE MATCH RATE FOR RECONSTRUCTIONS FROM ENCRYPTED TEMPLATES - please wait...")
decoder = initialize_onnx_session(cfg.decoder, use_cuda=True)
buffalo = initialize_onnx_session("../models/buffalo_l/w600k_r50.onnx", use_cuda=True)
cosines = []
for item in protected_identification_set_encrypted:
    if item['id'] in registration_db:  # we will check only templates with registration mate
        rec = reconstruct_face_by_decoder(decoder, item['emb'])
        if cfg.visualize:
            regphoto_filename = os.path.join(cfg.faces_dataset, registration_db[item['id']]['filename'])
            reg = fit_img_into_rectangle(cv2.imread(regphoto_filename, cv2.IMREAD_COLOR), 480, 480)
            cv2.imshow("reconstructed", rec)
            cv2.imshow("registration_db", reg)
            cv2.waitKey(cfg.visualize_ms)
        nt_from_synth_face = extract_template_from_synth_image(rec, buffalo, normalize=True)
        reg_template = registration_db[item['id']]['emb']
        cosine = np.dot(nt_from_synth_face, reg_template / np.linalg.norm(reg_template))
        cosines.append(cosine)
print_cosines_stats(cosines, threshold=cfg.buffalo_cosine_threshold)
print("CHECK POSITIVE MATCH RATE FOR RECONSTRUCTIONS FROM ENCRYPTED TEMPLATES - finished")