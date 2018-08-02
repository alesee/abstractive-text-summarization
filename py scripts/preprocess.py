from torchtext import data, vocab
from utils import BatchTuple
from fastai.dataset import ModelData

def load_data_build_vocab(path, src_col, trg_col, src_file, trg_file, pre_trained_vector_type):
    field = data.Field(tokenize='spacy', lower=True, eos_token='_eos_')
    data_fields = [(src_col, field), (trg_col, field)]
    trn, vld = data.TabularDataset.splits(path=f'{path}', train=src_file, validation=trg_file,
                                     format='csv', skip_header=True, fields=data_fields)
    if pre_trained_vector_type: 
        print('\n'+('='*100))
        print("Loading pretrained embeddings..... \n")
    field.build_vocab(trn, vectors=pre_trained_vector_type)
    return trn, vld, field
    
def batch_iterator(trn, vld, batch_size, src_col, trg_col,USE_GPU):
    train_iter, val_iter = data.BucketIterator.splits(
                        (trn, vld), batch_sizes=(batch_size,int(batch_size*1.6)),
                        device=(0 if USE_GPU else -1), 
                        sort_key=lambda x: len(getattr(x, src_col)), 
                        sort_within_batch=False, repeat=False)
    train_dl = BatchTuple(train_iter, src_col, trg_col)
    val_dl = BatchTuple(val_iter, src_col, trg_col)
    return train_dl, val_dl
    
def get_model_data(path, src_col, trg_col, src_file, trg_file, batch_size, USE_GPU, pre_trained_vector_type): 
    trn, vld, field = load_data_build_vocab(path, src_col, trg_col, src_file, trg_file, pre_trained_vector_type)
    train_dl, val_dl = batch_iterator(trn, vld, batch_size, src_col, trg_col, USE_GPU)
    model_data = ModelData(path, train_dl, val_dl)
    return field, model_data


