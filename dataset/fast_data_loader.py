import glob
import os
import time

import numpy as np
import pyarrow as pa
import torch

from loguru import logger
from torch.utils.data import Dataset


class MapfArrowDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, device, batch_size):
        self.all_data_files = self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8

        ddp_local_rank = os.environ.get("LOCAL_RANK")
        ddp_world_size = os.environ.get("WORLD_SIZE")
        # Divide files among DDP workers for training
        if "train" in folder_path and ddp_local_rank is not None and ddp_world_size is not None:
            ddp_local_rank, ddp_world_size = int(ddp_local_rank), int(ddp_world_size)
            files_per_worker = len(self.file_paths) // ddp_world_size
            start_index = ddp_local_rank * files_per_worker
            end_index = start_index + files_per_worker
            self.file_paths = self.file_paths[start_index:end_index]

        # pre-allocate memory for the input and target tensors (same file size)
        sample_input_tensors, sample_gt_actions = self._get_data_from_file(self.file_paths[0])

        self.input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(sample_input_tensors.shape, -1, dtype=self.dtype, device=self.device)

        logger.info(f"Single file tensor size: {self.input_tensors.numel() * self.input_tensors.element_size() / 1e9:.4f} GB")

    @staticmethod
    def _get_data_from_file(file_path):
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
            input_tensors = table["input_tensors"].to_numpy(zero_copy_only=False)
            gt_actions = table["gt_actions"].to_numpy(zero_copy_only=False)

        # shuffle data within the current file
        indices = np.random.permutation(len(input_tensors))
        input_tensors = np.stack(input_tensors[indices])
        gt_actions = gt_actions[indices]

        return input_tensors, gt_actions

    def load_and_transfer_data_file(self, filename):
        start_time = time.monotonic()

        input_tensors, gt_actions = self._get_data_from_file(filename)

        self.input_tensors.copy_(torch.tensor(input_tensors, dtype=self.dtype), non_blocking=True)
        self.target_tensors[:, -1].copy_(torch.tensor(gt_actions, dtype=self.dtype), non_blocking=True)
        finish_time = time.monotonic() - start_time
        logger.debug(f'Data from {filename} for {self.device} device prepared in ~{round(finish_time, 5)}s')

    def __iter__(self):
        while True:
            for file_path in self.file_paths:
                self.load_and_transfer_data_file(file_path)
                for i in range(0, len(self.input_tensors), self.batch_size):
                    yield self.input_tensors[i:i + self.batch_size], self.target_tensors[i:i + self.batch_size]

    def get_shard_size(self):
        return len(self.input_tensors) * len(self.file_paths)

    def get_full_dataset_size(self):
        return len(self.input_tensors) * len(self.all_data_files)


def main():
    # folder_path = "../dataset/validation"
    folder_path = "../dataset/train"
    dataset = MapfArrowDataset(folder_path, device='cuda:0', batch_size=32)
    data = iter(dataset)
    x = 0
    logger.info(dataset.get_full_dataset_size())
    logger.info(dataset.get_shard_size())

    while True:
        x += 1
        qx, qy = next(data)
        # logger.info(str(qx.shape) + ' ' + str(qy.shape))


if __name__ == "__main__":
    main()

