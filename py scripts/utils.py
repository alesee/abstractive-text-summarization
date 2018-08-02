import json
import os
import pickle
from fastai.sgdr import Callback, DecayScheduler
from fastai.model import Stepper
from fastai.learner import Learner
import torch.nn.functional as F
import torch.nn as nn


class BatchTuple():
    def __init__(self, dataset, x_var, y_var):
        self.dataset, self.x_var, self.y_var = dataset, x_var, y_var
        
    def __iter__(self):
        for batch in self.dataset:
            x = getattr(batch, self.x_var) 
            y = getattr(batch, self.y_var)                 
            yield (x, y)
            
    def __len__(self):
        return len(self.dataset)
    
    
class Seq2SeqStepper(Stepper):
    def step(self, xs, y, epoch):
        output = self.m(*xs, y)
        xtra = []
        if isinstance(output,tuple): output,*xtra = output
        self.opt.zero_grad()
        loss = raw_loss = self.crit(output, y)
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.clip:   # Gradient clipping
            nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        self.opt.step()
        return raw_loss.data[0]
            
class BestModelCheckPoint(Callback):
    def __init__(self, learner, path, model_name, lr):
        super().__init__()
        self.learner = learner
        self.model_name = model_name
        self.learning_rate = lr
        self.model_log = {}
        self.model_path = self.learner.models_path
        os.makedirs(self.model_path, exist_ok=True)

    def on_train_begin(self): 
        self.first_epoch = True
        self.epoch = 0
        self.best_loss = 0.

    def on_batch_begin(self): pass
    def on_phase_begin(self): pass
    def on_epoch_end(self, metrics): 
        self.epoch += 1
        self.val_loss = metrics[0]
        if self.first_epoch:
            self.best_loss = self.val_loss
            self.first_epoch = False
        elif self.val_loss < self.best_loss:
            self.best_loss = self.val_loss
            self.learner.save(self.model_name)
            self.model_log['training_loss'] = [str(self.train_losses)]
            self.model_log['validation_loss'] = [str(self.val_loss)]
            self.model_log['epoch_num'] = [str(self.epoch)]
            self.model_log['learning_rate'] = [str(self.learning_rate)]
            self.model_log['model_info'] = [w for s in [str(self.learner.model)] for w in s.split('\n')]
            self.model_log['model_info'].append("(attention_type): %s" %self.learner.model.attention_type)
            self.model_log['model_info'].append("(weight_tie): %s" %self.learner.model.tied_weight_type)
            self.model_log['model_info'].append("(pre_trained_vector_type): %s" %self.learner.model.pre_trained_vector_type)
            self.model_log['model_info'].append("(teacher_forcing): %s" %self.learner.model.teacher_forcing)
            if self.learner.model.teacher_forcing: self.model_log['model_info'].append("(teacher_forcing_prob): %s" %self.learner.model.force_prob)
            with open(f'{self.model_path}/{self.model_name}_model_log.json', 'w') as d: json.dump(self.model_log, d)
        else: pass        
    def on_phase_end(self): pass
    def on_batch_end(self, loss):
        self.train_losses = loss
    def on_train_end(self): pass

class TeacherForcingSched(Callback):
    def __init__(self, learner, scheduler):
        super().__init__()
        self.learner = learner
        self.scheduler = scheduler
        
    def on_train_begin(self): 
        self.learner.model.force_prob = round(self.scheduler.next_val(),1)
        
    def on_batch_begin(self): pass
    def on_phase_begin(self): pass
    def on_epoch_end(self, metrics): 
        self.learner.model.force_prob = round(self.scheduler.next_val(),1)
        
    def on_phase_end(self): pass
    def on_batch_end(self, loss):pass
    def on_train_end(self): pass
        
def save_pickle(path, filename, file):
    """Function to save file as pickle"""
    with open(f'{path}/{filename}', 'wb') as f:
        pickle.dump(file, f)
        
def load_pickle(path, filename):
    """Function to load pickle as file"""
    with open(f'{path}'+filename, 'rb') as file:
        output = pickle.load(file) 
    return output

        
def norm_pre_trained_embeddings(vecs, itos, em_sz, padding_idx):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=padding_idx)
    wgts = emb.weight.data
    for i,w in enumerate(itos):
        try: 
            wgts[i] = torch.from_numpy(vecs[w]-vec_mean)
            wgts[i] = torch.from_numpy(vecs[w]/vec_std)
        except: pass 
    emb.weight.requires_grad = False    
    return emb

def embedding_param(path, data_field, pre_trained_vector_type, embz_size=128):
    pre_trained=None
    index_to_string, string_to_index = data_field.vocab.itos, data_field.vocab.stoi
    vocab_path = os.path.join(path, "vocab")
    os.makedirs(vocab_path, exist_ok=True)
    save_pickle(vocab_path, 'itos.pk', index_to_string) 
    save_pickle(vocab_path, 'stoi.pk', string_to_index) 
    padding_idx = data_field.vocab.stoi['<pad>']
    if pre_trained_vector_type:
        vec_mean, vec_std = data_field.vocab.vectors.numpy().mean(), data_field.vocab.vectors.numpy().std()
        vector_weight_matrix = data_field.vocab.vectors
        embz_size = vector_weight_matrix.size(1)
        pre_trained = norm_pre_trained_embeddings(vector_weight_matrix, index_to_string, embz_size, padding_idx)
    return pre_trained,  embz_size, padding_idx

def seq2seq_loss(input, target):
    sl,bs = target.size()
    sl_in,bs_in,nc = input.size()
    if sl>sl_in: input = F.pad(input, (0,0,0,0,0,sl-sl_in))
    input = input[:sl]
    return F.cross_entropy(input.view(-1,nc), target.view(-1))
        
