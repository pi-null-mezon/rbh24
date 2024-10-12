import numpy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tools import make_onnx_inference, tensor2image, image2tensor
from tqdm import tqdm
import numpy as np
import tqdm
import cv2
import os
import sys


class CustomDataset(Dataset):
    def __init__(self, photos_path,  max_ids):
        max_ids = float('inf') if max_ids == -1 else max_ids
        self.photos_path = photos_path
        self.filenames = []
        self.isreg = []
        self.labels = []
        ids_collected = 0
        for label in [s.name for s in os.scandir(self.photos_path) if s.is_dir()]:
            samples_for_label_collected = 0
            for filename in [f.name for f in os.scandir(os.path.join(self.photos_path, label)) if f.is_file()]:
                self.filenames.append(os.path.join(label, filename))
                self.isreg.append(bool(samples_for_label_collected == 0))
                self.labels.append(label)
                samples_for_label_collected += 1
            ids_collected += 1
            if ids_collected > max_ids:
                break

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.photos_path, self.filenames[idx])
        mat = cv2.imread(filename, cv2.IMREAD_COLOR)
        return filename, self.isreg[idx], self.labels[idx], mat


def calculate_identifications(iset, regdb, top_k):
    top = []
    print("IDENTIFICATION - please wait...")
    for it in tqdm.tqdm(iset):
        predictions = []
        for _id, rt in regdb.items():
            cosine = np.dot(it['emb'] / np.linalg.norm(it['emb']), rt['emb'] / np.linalg.norm(rt['emb']))
            predictions.append({"cos": cosine, "id": _id})
        predictions.sort(key=lambda x: x['cos'], reverse=True)
        top.append({'id': it['id'], 'predictions': predictions[:top_k]})
    print("IDENTIFICATION - finished")
    return top


def prob_estimation(positive_outcomes, total_outcomes, reliable_threshold=3):
    if total_outcomes < reliable_threshold:
        return None
    if reliable_threshold < positive_outcomes <= (total_outcomes - reliable_threshold):
        return positive_outcomes / total_outcomes
    elif positive_outcomes > (total_outcomes - reliable_threshold):
        return (total_outcomes - reliable_threshold) / total_outcomes
    return reliable_threshold / total_outcomes


def calculate_curves(matches, steps_total=1E4, max_rank_for_fnir=1, epsilon=1.0E-6):
    progress_bar = tqdm.tqdm(total=len(matches), file=sys.stdout)
    mate_found_list = []
    non_mates_list = []
    for match in matches:
        rank = 1
        for prediction in match['predictions']:
            if prediction['id'] == match['id']:
                mate_found_list.append({'rank': rank, 'similarity': prediction['cos']})
            else:
                non_mates_list.append(prediction['cos'])
            rank += 1
        progress_bar.update(1)
    progress_bar.close()

    progress_bar = tqdm.tqdm(total=int(steps_total), file=sys.stdout)
    roc = []
    numpy_non_mates = np.asarray(non_mates_list)
    for similarity_threshold in np.linspace(start=-1.0 - epsilon, stop=1.0 + epsilon, num=int(steps_total)):
        false_negative_identifications = 0
        for item in mate_found_list:
            if item['similarity'] < similarity_threshold or item['rank'] > max_rank_for_fnir:
                false_negative_identifications += 1
        false_positive_identifications = np.sum(numpy_non_mates >= similarity_threshold)
        roc.append({'cos': similarity_threshold,
                    'fnir': prob_estimation(false_negative_identifications, len(mate_found_list)),
                    'fpir': prob_estimation(false_positive_identifications, len(numpy_non_mates))})
        progress_bar.update(1)
    progress_bar.close()
    return roc


def find_fnir(target_fpir, roc):
    if roc[0]['fpir'] is None:
        return None
    for i in range(0, len(roc)-1):
        if (roc[i]['fpir'] - target_fpir)*(roc[i+1]['fpir'] - target_fpir) <= 0:
            return roc[i]['fnir'] + (roc[i]['fpir'] - target_fpir) * \
                   (roc[i+1]['fnir'] - roc[i]['fnir']) / (roc[i]['fpir'] - roc[i+1]['fpir'])
    return roc[len(roc)-1]['fnir']


def draw_rocs(rocs, title, labels, axis_limits=[1E-4, 1E-0, 1E-2, 1E-0]):
    linestyles_list = ['-', '--', '-.', ':']
    for i, roc in enumerate(rocs):
        fpir, fnir = [], []
        for point in roc:
            fpir.append(point['fpir'])
            fnir.append(point['fnir'])
        plt.plot(fpir, fnir, label=labels[i], linestyle=linestyles_list[i % len(linestyles_list)])
    plt.legend(loc="upper right")
    plt.ylabel('FNIR')
    plt.yscale('log')
    plt.xlabel('FPIR')
    plt.xscale('log')
    plt.title(title)
    plt.grid(True, which='both', alpha=0.4)
    fig = plt.gcf()
    fig.set_size_inches(14, 10)
    plt.axis(axis_limits)
    plt.show()


def convert_keypoints_to_bbox_normalized(keypoints, bbox):
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    normalized_keypoints = []
    for kp in keypoints:
        x, y = kp
        x_bbox = x - x1
        y_bbox = y - y1
        x_norm = x_bbox / bbox_width
        y_norm = y_bbox / bbox_height
        normalized_keypoints.append([x_norm, y_norm])
    return 1000 * np.array(normalized_keypoints)

'''
def generate_key_from_keypoints(keypoints, length):
    keypoints_bytes = keypoints.tobytes()
    key = np.frombuffer(keypoints_bytes, dtype=np.uint8)
    return np.resize(key, length)


def encrypt_template(biometric_vector, keypoints):
    biometric_bytes = np.frombuffer(biometric_vector.tobytes(), dtype=np.uint8)
    key = generate_key_from_keypoints(keypoints, len(biometric_bytes))
    encrypted_data = np.bitwise_xor(biometric_bytes, key)
    return np.frombuffer(encrypted_data.tobytes(), dtype=np.float32)


def decrypt_template(encrypted_vector, keypoints):
    biometric_bytes = np.frombuffer(encrypted_vector.tobytes(), dtype=np.uint8)
    key = generate_key_from_keypoints(keypoints, len(biometric_bytes))
    decrypted_data = np.bitwise_xor(biometric_bytes, key)
    return np.frombuffer(decrypted_data.tobytes(), dtype=np.float32)
'''


def generate_key_from_keypoints(normalized_keypoints, length):
    return 1000 * np.resize(normalized_keypoints.flatten(), length)  # multiply to be greater than 1.


def encrypt_template(biometric_vector, normalized_keypoints):
    key = generate_key_from_keypoints(normalized_keypoints, biometric_vector.shape[0])
    encrypted_data = (biometric_vector + key) / key
    return encrypted_data


def decrypt_template(encrypted_vector, normalized_keypoints):
    key = generate_key_from_keypoints(normalized_keypoints, encrypted_vector.shape[0])
    decrypted_data = encrypted_vector * key - key
    return decrypted_data


def reconstruct_face_by_decoder(decoder, template):
    nt = template / np.linalg.norm(template)
    if nt.shape == (512,):
        nt = numpy.reshape(nt, (1, 512))
    rec = make_onnx_inference(session=decoder, input_data=nt)
    rec = tensor2image(rec.squeeze(0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], swap_red_blue=True)
    return rec


def extract_template_from_synth_image(img, buffalo_onnx_session, normalize):
    resized_img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    tensor = image2tensor(resized_img, mean=3 * [127.5 / 255], std=3 * [127.5 / 255], swap_red_blue=True)
    tensor = np.expand_dims(tensor, axis=0)
    template = make_onnx_inference(buffalo_onnx_session, tensor)
    if normalize:
        template = template / np.linalg.norm(template)
    return template


def print_cosines_stats(cosines, threshold):
    print(f"STATISTICS ON {len(cosines)} SAMPLES:")
    cosines = np.array(cosines)
    print(f" - COSINE MIN:    {cosines.min().item():.4f}")
    print(f" - COSINE MEAN:   {cosines.mean().item():.4f}")
    print(f" - COSINE MEDIAN: {np.median(cosines).item():.4f}")
    print(f" - COSINE MAX:    {cosines.max().item():.4f}")
    tp = np.sum(cosines > threshold)
    print(
        f"TOTAL: {tp} of {len(cosines)} have cosine with genuine template greater than {threshold:.3f}"
        f" >> it is {100 * tp / len(cosines):.1f} % of samples")
