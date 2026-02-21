compile = True
max_iters = 400000
lr_decay_iters = 400000

batch_size = 512
n_layer = 12
n_head = 12
n_embd = 768

train_data_file = "dataset/train"
valid_data_file = "dataset/validation"

block_size = 256
gradient_accumulation_steps = 16

# init_from = 'resume'
