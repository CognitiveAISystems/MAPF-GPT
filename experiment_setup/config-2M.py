compile = True
max_iters = 30000
lr_decay_iters = 30000

batch_size = 4096
n_layer = 5
n_head = 5
n_embd = 160

train_data_file = "dataset/train"
valid_data_file = "dataset/validation"

block_size = 256
gradient_accumulation_steps = 16

# init_from = 'resume'
