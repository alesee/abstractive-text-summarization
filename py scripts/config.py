import torch
USE_GPU = torch.cuda.is_available()
DATA_PATH = 'data/'
SAMPLE_DATA_PATH = f'{DATA_PATH}sample_data/'
path = SAMPLE_DATA_PATH
src_col = 'source'
trg_col = 'target'
src_file = 'train_ds.csv'
trg_file = 'valid_ds.csv'
batch_size = 32
max_tgt_len = 25
hidden_size = 512
rnn_type = 'gru'
attention_type='luong'
tied_weight_type ='three_way'
pre_trained_vector_type = None
lr=1e-3
epochs = 5
