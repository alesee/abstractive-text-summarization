from config import *
from fastai.nlp import *
from utils import *
from preprocess import *
from model import Seq2SeqRNN 


def get_model():
    TEXT, model_data = get_model_data(path, src_col, trg_col, src_file, trg_file, batch_size, USE_GPU, pre_trained_vector_type)
    input_size = output_size= len(TEXT.vocab)
    pre_trained_vector,  embz_size, padding_idx = embedding_param(path, TEXT, pre_trained_vector_type)
    model = Seq2SeqRNN(rnn_type, input_size, embz_size, hidden_size, batch_size, output_size, max_tgt_len,
               attention_type, tied_weight_type, pre_trained_vector, pre_trained_vector_type, padding_idx)
    print('='*100)
    print('Model log:')
    print(model, '\n')
    print('- attention_type = {} \n'.format(model.attention_type))
    print('- weight_tie = {} \n'.format(model.tied_weight_type))
    print('- teacher_forcing = {} \n '.format(model.teacher_forcing)) 
    print('- pre_trained_embedding = {} \n'.format(model.pre_trained_vector_type)) 
    print('='*100 + '\n')
    return model_data, model

def main(model_data, model):
    if USE_GPU: model.cuda()
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    learn = RNN_Learner(model_data, SingleModel(model), opt_fn=opt_fn)
    learn.crit = seq2seq_loss
    model_name = f'{model.rnn_type}_{model.attention_type}'.lower()
    best_model = BestModelCheckPoint(learn, path, model_name, lr)
    sched = DecayScheduler(DecayType.LINEAR, epochs, 0.6, 0.3)
    teach_forcer = TeacherForcingSched(learn, sched)
    print("\nTraining......\n")
    learn.fit(lr, 1, cycle_len=epochs, use_clr=(20,10), stepper=Seq2SeqStepper, callbacks=[teach_forcer, best_model])
    print("\nTraining complete......\n")


if __name__ == '__main__':
    model_data, model = get_model()
    main(model_data, model)
    
