from torch.utils.data import Dataset
import onnxruntime as ort
import numpy as np
import cv2
import os


def image2tensor(bgr, mean, std, swap_red_blue=False):
    tmp = bgr.astype(dtype=np.float32) / 255.0
    if swap_red_blue:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    tmp = np.transpose(tmp, axes=(2, 0, 1))  # HxWxC -> CxHxW
    tmp -= np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    tmp /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return tmp


class CustomDataSet(Dataset):
    def __init__(self, images_path, start_id=0, max_ids=-1, mean=3*[127.5/255], std=3*[127.5/255], swap_red_blue=True):
        self.swap_red_blue = swap_red_blue
        self.mean = mean
        self.std = std
        self.targets = []
        self.files = []
        self.images_path = images_path
        dirs_list = [s.name for s in os.scandir(self.images_path) if s.is_dir()]
        dirs_list.sort()
        self.start_id = start_id
        self.max_ids = len(dirs_list) if max_ids == -1 else max_ids
        for i, subdir in enumerate(dirs_list[self.start_id:(self.start_id + self.max_ids)]):
            for filename in [f.name for f in os.scandir(os.path.join(self.images_path, subdir)) if f.is_file()]:
                self.files.append(os.path.join(subdir, filename))
                self.targets.append(i)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.targets[idx]
        filename = self.files[idx]
        sample = cv2.imread(os.path.join(self.images_path, filename), cv2.IMREAD_COLOR)
        normalized_sample = image2tensor(sample, mean=self.mean, std=self.std, swap_red_blue=self.swap_red_blue)
        return label, filename, normalized_sample


def initialize_onnx_session(model_path, use_cuda=True):
    # Set up the execution providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']

    # Create session options
    sess_options = ort.SessionOptions()

    # Optional: You can set some session options here
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads = 1

    try:
        # Create ONNX Runtime session
        session = ort.InferenceSession(model_path, sess_options, providers=providers)

        # Check if CUDA is actually being used
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


def normalize_l2(matrix):
    """
    Normalize each row of the matrix to have unit L2 norm.

    Args:
    matrix (numpy.ndarray): Input 2D matrix to be normalized.

    Returns:
    numpy.ndarray: Matrix with L2 normalized rows.
    """
    # Compute L2 norm along axis 1 (rows)
    row_norms = np.linalg.norm(matrix, axis=1)

    # Reshape row_norms to allow broadcasting
    row_norms = row_norms[:, np.newaxis]

    # Divide each row by its norm
    normalized_matrix = matrix / row_norms

    return normalized_matrix