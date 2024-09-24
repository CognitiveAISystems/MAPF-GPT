from huggingface_hub import hf_hub_download

repo_id = "aandreychuk/MAPF-GPT"
local_dir = "dataset"

# Download the 'validation' part
hf_hub_download(repo_id=repo_id, repo_type='dataset', subfolder="validation", filename="chunk_0_part_0.arrow", local_dir=local_dir)

# Download the 'train' part
# reduce the number of chunks or parts if you don't need the whole dataset
# each file contains 2**21 input tensors and requires 512 MB of disk space
for chunk in range(50):
    for part in range(10):
        hf_hub_download(repo_id=repo_id, repo_type='dataset', subfolder="train", filename=f"chunk_{chunk}_part_{part}.arrow", local_dir=local_dir)
