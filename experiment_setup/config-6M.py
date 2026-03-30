compile = True
max_iters = 30000
lr_decay_iters = 30000

batch_size = 2048
n_layer = 8
n_head = 8
n_embd = 256

train_data_file = "dataset/train"
valid_data_file = "dataset/validation"

block_size = 256
gradient_accumulation_steps = 16

# init_from = 'resume'
