from time import perf_counter
from torch.utils.data import Dataset
import albumentations as A
import onnxruntime as ort
import pickle
import numpy as np
import torch
import cv2
import os
import sys
from skimage import transform as trans



def normalize_image(bgr, mean, std, swap_red_blue=False):
    tmp = bgr.astype(dtype=np.float32) / 255.0
    if swap_red_blue:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    tmp -= np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    tmp /= np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return tmp


def image2tensor(bgr, mean, std, swap_red_blue=False):
    tmp = normalize_image(bgr, mean, std, swap_red_blue)
    return np.transpose(tmp, axes=(2, 0, 1))  # HxWxC -> CxHxW


def tensor2image(tensor, mean, std, swap_red_blue=False):
    tmp = np.transpose(tensor, axes=(1, 2, 0))  # CxHxW -> HxWxC
    tmp *= np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    tmp += np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    if swap_red_blue:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    tmp *= 255.0
    return tmp.astype(dtype=np.uint8)


class CustomDataSet(Dataset):
    def __init__(self, templates_paths, photos_path, do_aug, size, mean, std, swap_reb_blue, normalize_templates,
                 max_samples_per_id=-1):
        assert max_samples_per_id != 0, "max_samples_per_id should not be equal to 0"
        self.mean = mean
        self.std = std
        self.swap_red_blue = swap_reb_blue
        self.size = size
        self.do_aug = do_aug
        self.photos_path = photos_path
        self.normalize_templates = normalize_templates
        self.templates = []
        self.filenames = []
        for filename in templates_paths:
            with open(filename, 'rb') as i_f:
                data = pickle.load(i_f)
                if max_samples_per_id == -1:
                    self.filenames += data['file']
                    self.templates += data['buffalo']
                else:
                    tmp = {}
                    for filename, template in zip(data['file'], data['buffalo']):
                        _id = filename.split('/', 1)[0]
                        if _id not in tmp:
                            tmp[_id] = [(filename, template)]
                        elif len(tmp[_id]) < max_samples_per_id:
                            tmp[_id].append((filename, template))
                    for _id in tmp:
                        for filename, template in tmp[_id]:
                            self.filenames.append(filename)
                            self.templates.append(template)
        self.album = A.Compose([
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.HorizontalFlip(p=0.5),
            A.Affine(p=1.0, scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-5, 5)),
            A.GaussNoise(p=0.5, var_limit=(1, 15)),
            A.ImageCompression(p=0.25, quality_lower=80, quality_upper=100),
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.ColorJitter(p=0.5)
        ], p=1.0)

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, idx):
        filename = os.path.join(self.photos_path, self.filenames[idx])
        mat = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.do_aug:
            mat = self.album(image=mat)["image"]
        if mat.shape[0] != self.size[1] and mat.shape[1] != self.size[0]:
            interp = cv2.INTER_LINEAR if mat.shape[0]*mat.shape[1] > self.size[0]*self.size[1] else cv2.INTER_CUBIC
            mat = cv2.resize(mat, self.size, interpolation=interp)
        # Visual control
        # cv2.imshow("probe", mat)
        # cv2.waitKey(0)
        template = self.templates[idx]
        if self.normalize_templates:
            template = template / np.linalg.norm(template)
        return torch.from_numpy(template),\
            torch.from_numpy(image2tensor(mat, mean=self.mean, std=self.std, swap_red_blue=self.swap_red_blue))


class Speedometer:
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        self._speed = None
        self.t0 = perf_counter()

    def reset(self):
        self._speed = None
        self.t0 = perf_counter()

    def update(self, samples):
        if self._speed is None:
            self._speed = samples / (perf_counter() - self.t0)
        else:
            self._speed = self._speed * self.gamma + (1 - self.gamma) * samples / (perf_counter() - self.t0)
        self.t0 = perf_counter()

    def speed(self):
        return self._speed


class Averagemeter:
    def __init__(self, gamma: float = 0.999):
        self.gamma = gamma
        self.val = None

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.val * self.gamma + (1 - self.gamma) * val

    def value(self):
        return self.val


def print_one_line(s):
    sys.stdout.write('\r' + s)
    sys.stdout.flush()


def model_size_mb(model):
    params_size = 0
    for param in model.parameters():
        params_size += param.nelement() * param.element_size()
    buffers_size = 0
    for buffer in model.buffers():
        buffers_size += buffer.nelement() * buffer.element_size()
    return (params_size + buffers_size) / 1024 ** 2


def read_img_as_torch_tensor(filename, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), swap_rb=True):
    mat = cv2.imread(filename, cv2.IMREAD_COLOR)
    assert (mat.shape[0] == size[1] and mat.shape[1] == size[0]), "sizes missmatch"
    return torch.from_numpy(image2tensor(mat, mean=mean, std=std, swap_red_blue=swap_rb))


def initialize_onnx_session(model_path, use_cuda=True):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    try:
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        if use_cuda:
            assert 'CUDAExecutionProvider' in session.get_providers(), "CUDA is not available"
        print(f"ONNX Runtime session initialized with providers: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"Error initializing ONNX Runtime session: {str(e)}")
        return None


def make_onnx_inference(session, input_data):
    results = session.run(None, {session.get_inputs()[0].name: input_data})
    return results[0]


def extract_template_from_image(image, fa_model):
    info = fa_model.get(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if len(info) > 0:
        info = sorted(info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # biggest one
        return info
    return None


arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def extract_template_from_synth_image(img, buffalo_onnx_session, normalize):
    resized_img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    tensor = image2tensor(resized_img, mean=3 * [127.5 / 255], std=3 * [127.5 / 255], swap_red_blue=True)
    tensor = np.expand_dims(tensor, axis=0)
    template = make_onnx_inference(buffalo_onnx_session, tensor)
    if normalize:
        template = template / np.linalg.norm(template)
    return template


def torch2numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()