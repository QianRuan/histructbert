import logging
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import BertConfig, BertModel, BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertLayerNorm

logger = logging.getLogger(__name__)

HiStructBert_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    }

HiStructBert_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
    }
def sin_positional_encoding(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

#class SentSinPositionalEncoding(nn.Module):
#
#    def __init__(self, dropout, dim, max_len=5000):
#        pe = torch.zeros(max_len, dim)
#        
#        position = torch.arange(0, max_len).unsqueeze(1)
#        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
#                              -(math.log(10000.0) / dim)))
#        pe[:, 0::2] = torch.sin(position.float() * div_term)
#        pe[:, 1::2] = torch.cos(position.float() * div_term)
#        pe = pe.unsqueeze(0)
#        super(SentSinPositionalEncoding, self).__init__()
#        self.register_buffer('pe', pe)
#        self.dropout = nn.Dropout(p=dropout)
#        self.dim = dim
#
#    def forward(
#        self,
#        top_vecs,
#        tok_struct_vec,
#        sent_struct_vec,
#        position_ids=None,
#        step=None
#    ):
#        #emb = emb * math.sqrt(self.dim)
#        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
#        
#        if (step):
#            #emb = emb + self.pe[:, step][:, None, :]
#            pe = self.pe[:, step][:, None, :]
#
#        else:
##            emb = emb + self.pe[:, :emb.size(1)]
#            pe = self.pe[:, :n_sents]
#        pe = self.dropout(pe)
#        return pe

class SinPositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(SinPositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(
        self,
        inputs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
#        step=None
    ):
        #emb = emb * math.sqrt(self.dim)
        batch_size = inputs.size(0)
        n = inputs.size(1)
        pe = self.pe[:, :n]
        
        pos_embs = pe.expand(batch_size,-1,-1)
        
#        if (step):
#            #emb = emb + self.pe[:, step][:, None, :]
#            pe = self.pe[:, step][:, None, :]
#
#        else:
#            emb = emb + self.pe[:, :emb.size(1)]
#            pe = self.pe[:, :n_tokens]
        #pe = pe.expand(batch_size,-1,-1)
        #pe = self.dropout(pe)
        #print(pe[0])
        #print(pe[1])
        
        
        return pe,pos_embs

class HiStructBertConfig(BertConfig):
    pretrained_config_archive_map = HiStructBert_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "bert"

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        #self.max_2d_position_embeddings = max_2d_position_embeddings
        
class SINTokInputEmbSUM(nn.Module):
    def __init__(self, config):
        super(SINTokInputEmbSUM, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = SinPositionalEncoding(
            config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )      
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        pe, position_embeddings= self.position_embeddings(input_ids,tok_struct_vec,sent_struct_vec)
        
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        
        def compute_se(pos, pe, position_embeddings):       
            x_position_embeddings = torch.zeros_like(position_embeddings)
    #        print("########pe[0][10]",pe[0][10].shape,pe[0][10])
            for i in range(pos.size(0)):
    #            print(para_pos.size(0))
                for j in range (pos.size(1)):
    #                print(para_pos.size(1))
    #                print(para_pos[i][j])
                    x_position_embeddings[i][j] = pe[0][para_pos[i][j]]
            return x_position_embeddings
                    
        
        para_position_embeddings = compute_se(para_pos, pe, position_embeddings)
        sent_position_embeddings = compute_se(sent_pos, pe, position_embeddings)
        tok_position_embeddings = compute_se(tok_pos, pe, position_embeddings)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########para_pos[0][100]",para_pos[0][100].shape,para_pos[0][100])
#        print("########pe[0][para_pos[0][100]]",pe[0][para_pos[0][100]].shape,pe[0][para_pos[0][100]])
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        
        tok_struct_embeddings = para_position_embeddings+sent_position_embeddings+tok_position_embeddings

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings   
     
class SINTokInputEmbMEAN(nn.Module):
    def __init__(self, config):
        super(SINTokInputEmbMEAN, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = SinPositionalEncoding(
            config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )      
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        pe,position_embeddings = self.position_embeddings(input_ids,tok_struct_vec,sent_struct_vec)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        
        def compute_se(pos, pe, position_embeddings):       
            x_position_embeddings = torch.zeros_like(position_embeddings)
    #        print("########pe[0][10]",pe[0][10].shape,pe[0][10])
            for i in range(pos.size(0)):
    #            print(para_pos.size(0))
                for j in range (pos.size(1)):
    #                print(para_pos.size(1))
    #                print(para_pos[i][j])
                    x_position_embeddings[i][j] = pe[0][para_pos[i][j]]
            return x_position_embeddings
                    
        
        para_position_embeddings = compute_se(para_pos, pe, position_embeddings)
        sent_position_embeddings = compute_se(sent_pos, pe, position_embeddings)
        tok_position_embeddings = compute_se(tok_pos, pe, position_embeddings)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########para_pos[0][100]",para_pos[0][100].shape,para_pos[0][100])
#        print("########pe[0][para_pos[0][100]]",pe[0][para_pos[0][100]].shape,pe[0][para_pos[0][100]])
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        
        tok_struct_embeddings = (para_position_embeddings+sent_position_embeddings+tok_position_embeddings)/3

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings        
    
class SINTokInputEmbCONCAT(nn.Module):
    def __init__(self, config):
        super(SINTokInputEmbCONCAT, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = SinPositionalEncoding(
            config.hidden_size
        )
        
        self.histruct_position_embeddings = SinPositionalEncoding(
            int(config.hidden_size/3)
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )      
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        #768 dim
        pe, position_embeddings= self.position_embeddings(input_ids,tok_struct_vec,sent_struct_vec)
#        position_embeddings = pe.expand(batch_size,-1,-1)
        
        #256 dim
        hs_pe,hs_position_embeddings = self.histruct_position_embeddings(input_ids,tok_struct_vec,sent_struct_vec)
#        hs_position_embeddings = hs_pe.expand(batch_size,-1,-1)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        
        def compute_se(pos, pe, position_embeddings):       
            x_position_embeddings = torch.zeros_like(position_embeddings)
    #        print("########pe[0][10]",pe[0][10].shape,pe[0][10])
            for i in range(pos.size(0)):
    #            print(para_pos.size(0))
                for j in range (pos.size(1)):
    #                print(para_pos.size(1))
    #                print(para_pos[i][j])
                    x_position_embeddings[i][j] = pe[0][para_pos[i][j]]
            return x_position_embeddings
                    
        
        para_position_embeddings = compute_se(para_pos, hs_pe, hs_position_embeddings)
        sent_position_embeddings = compute_se(sent_pos, hs_pe, hs_position_embeddings)
        tok_position_embeddings = compute_se(tok_pos, hs_pe, hs_position_embeddings)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########para_pos[0][100]",para_pos[0][100].shape,para_pos[0][100])
#        print("########pe[0][para_pos[0][100]]",pe[0][para_pos[0][100]].shape,pe[0][para_pos[0][100]])
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        
        #CONCAT
        tok_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings,tok_position_embeddings),2)

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings        
        
class LPTokInputEmbSUM(nn.Module):
    def __init__(self, config):
        super(LPTokInputEmbSUM, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )      
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.position_embeddings(para_pos)
        sent_position_embeddings = self.position_embeddings(sent_pos)
        tok_position_embeddings = self.position_embeddings(tok_pos)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        tok_struct_embeddings = para_position_embeddings+sent_position_embeddings+tok_position_embeddings

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class LPTokInputEmbMEAN(nn.Module):
    def __init__(self, config):
        super(LPTokInputEmbMEAN, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.position_embeddings(para_pos)
        sent_position_embeddings = self.position_embeddings(sent_pos)
        tok_position_embeddings = self.position_embeddings(tok_pos)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        
        #MEAN
        tok_struct_embeddings = (para_position_embeddings+sent_position_embeddings+tok_position_embeddings)/3
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########tok_struct_embeddings",tok_struct_embeddings.shape,tok_struct_embeddings)
        
        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class LATokInputEmb(nn.Module):
    def __init__(self, config, args):
        super(LATokInputEmb, self).__init__()
        self.args=args
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        
        self.a_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.b_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.c_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        tok_position_embeddings = self.c_position_embeddings(tok_pos)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        
        if(self.args.tok_se_comb_mode == 'sum'):
            tok_struct_embeddings = para_position_embeddings+sent_position_embeddings+tok_position_embeddings
        if(self.args.tok_se_comb_mode == 'mean'):
            tok_struct_embeddings = (para_position_embeddings+sent_position_embeddings+tok_position_embeddings)/3
        if(self.args.tok_se_comb_mode == 'concat'):
            tok_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings,tok_position_embeddings),2)
            

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    


class LATokInputEmbSUM(nn.Module):
    def __init__(self, config):
        super(LATokInputEmbSUM, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        
        self.a_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.b_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.c_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        tok_position_embeddings = self.c_position_embeddings(tok_pos)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        tok_struct_embeddings = para_position_embeddings+sent_position_embeddings+tok_position_embeddings

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class LATokInputEmbMEAN(nn.Module):
    def __init__(self, config):
        super(LATokInputEmbMEAN, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        
        self.a_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.b_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.c_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        tok_position_embeddings = self.c_position_embeddings(tok_pos)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        
        #MEAN
        tok_struct_embeddings = (para_position_embeddings+sent_position_embeddings+tok_position_embeddings)/3
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########tok_struct_embeddings",tok_struct_embeddings.shape,tok_struct_embeddings)
        
        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class LATokInputEmbCONCAT(nn.Module):
    def __init__(self, config):
        super(LATokInputEmbCONCAT, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        
        self.a_position_embeddings = nn.Embedding(
            config.max_position_embeddings, int((config.hidden_size)/3)
        )
        self.b_position_embeddings = nn.Embedding(
            config.max_position_embeddings, int((config.hidden_size)/3)
        )
        self.c_position_embeddings = nn.Embedding(
            config.max_position_embeddings, int((config.hidden_size)/3)
        )
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = tok_struct_vec[:,:,0]
        sent_pos = tok_struct_vec[:,:,1]
        tok_pos = tok_struct_vec[:,:,2]
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        tok_position_embeddings = self.c_position_embeddings(tok_pos)
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        
        tok_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings,tok_position_embeddings),2)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########tok_struct_embeddings",tok_struct_embeddings.shape,tok_struct_embeddings)
        

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + tok_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class SINSentAddEmbSUM(nn.Module):
    def __init__(self, config, max_nsent):
       
        super(SINSentAddEmbSUM, self).__init__()
    
        self.position_embeddings = SinPositionalEncoding(
            config.hidden_size
        )
       
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        
        pe, position_embeddings = self.position_embeddings(top_vecs,tok_struct_vec,sent_struct_vec)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        def compute_se(pos, pe, position_embeddings):       
            x_position_embeddings = torch.zeros_like(position_embeddings)
    #        print("########pe[0][10]",pe[0][10].shape,pe[0][10])
            for i in range(pos.size(0)):
    #            print(para_pos.size(0))
                for j in range (pos.size(1)):
    #                print(para_pos.size(1))
    #                print(para_pos[i][j])
                    x_position_embeddings[i][j] = pe[0][para_pos[i][j]]
            return x_position_embeddings
                    
        
        para_position_embeddings = compute_se(para_pos, pe, position_embeddings)
        sent_position_embeddings = compute_se(sent_pos, pe, position_embeddings)
        
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        #SUM
        sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    
class SINSentAddEmbMEAN(nn.Module):
    def __init__(self, config, max_nsent):
       
        super(SINSentAddEmbMEAN, self).__init__()
    
        self.position_embeddings = SinPositionalEncoding(
            config.hidden_size
        )
       
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        
        pe, position_embeddings = self.position_embeddings(top_vecs,tok_struct_vec,sent_struct_vec)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        def compute_se(pos, pe, position_embeddings):       
            x_position_embeddings = torch.zeros_like(position_embeddings)
    #        print("########pe[0][10]",pe[0][10].shape,pe[0][10])
            for i in range(pos.size(0)):
    #            print(para_pos.size(0))
                for j in range (pos.size(1)):
    #                print(para_pos.size(1))
    #                print(para_pos[i][j])
                    x_position_embeddings[i][j] = pe[0][para_pos[i][j]]
            return x_position_embeddings
                    
        
        para_position_embeddings = compute_se(para_pos, pe, position_embeddings)
        sent_position_embeddings = compute_se(sent_pos, pe, position_embeddings)
        
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        #SUM
        sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
class SINSentAddEmbCONCAT(nn.Module):
    def __init__(self, config, max_nsent):
       
        super(SINSentAddEmbCONCAT, self).__init__()
    
        self.position_embeddings = SinPositionalEncoding(
            config.hidden_size
        )
        
        self.histruct_position_embeddings = SinPositionalEncoding(
            int(config.hidden_size/2)
        )
       
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        #768 dim
        pe, position_embeddings = self.position_embeddings(top_vecs,tok_struct_vec,sent_struct_vec)
        #768/2 dim
        hs_pe,hs_position_embeddings = self.histruct_position_embeddings(top_vecs,tok_struct_vec,sent_struct_vec)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        def compute_se(pos, pe, position_embeddings):       
            x_position_embeddings = torch.zeros_like(position_embeddings)
    #        print("########pe[0][10]",pe[0][10].shape,pe[0][10])
            for i in range(pos.size(0)):
    #            print(para_pos.size(0))
                for j in range (pos.size(1)):
    #                print(para_pos.size(1))
    #                print(para_pos[i][j])
                    x_position_embeddings[i][j] = pe[0][para_pos[i][j]]
            return x_position_embeddings
                    
        
        para_position_embeddings = compute_se(para_pos, hs_pe, hs_position_embeddings)
        sent_position_embeddings = compute_se(sent_pos, hs_pe, hs_position_embeddings)
        
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        #SUM
        sent_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings),2)
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
class LASentAddEmbSUM(nn.Module):
    def __init__(self, config, max_nsent):
       
        super(LASentAddEmbSUM, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
       
        self.a_position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
        self.b_position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        self.add_embeddings = embeddings
        return embeddings

        

class LASentAddEmbMEAN(nn.Module):
    def __init__(self, config, max_nsent):
       
        super(LASentAddEmbMEAN, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
       
        self.a_position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
        self.b_position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        self.add_embeddings = embeddings
        return embeddings
    
class LASentAddEmbCONCAT(nn.Module):
     def __init__(self, config, max_nsent):
       
        super(LASentAddEmbCONCAT, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
       
        self.a_position_embeddings = nn.Embedding(
            max_nsent, int(config.hidden_size/2)
        )
        self.b_position_embeddings = nn.Embedding(
            max_nsent, int(config.hidden_size/2)
        )
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

     def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        
        para_position_embeddings = self.a_position_embeddings(para_pos)
        sent_position_embeddings = self.b_position_embeddings(sent_pos)
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        sent_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings),2)
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        self.add_embeddings = embeddings
        return embeddings
    


    
    
class LPSentAddEmbPOS(nn.Module):
     def __init__(self, config, max_nsent):
       
        super(LPSentAddEmbPOS, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size/2
        )
              
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

     def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        
        return position_embeddings
    
class LPSentAddEmbSUM(nn.Module):
    def __init__(self, config, max_nsent):
       
        super(LPSentAddEmbSUM, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
       
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        
        para_position_embeddings = self.position_embeddings(para_pos)
        sent_position_embeddings = self.position_embeddings(sent_pos)
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

        

class LPSentAddEmbMEAN(nn.Module):
    def __init__(self, config, max_nsent):
       
        super(LPSentAddEmbMEAN, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            max_nsent, config.hidden_size
        )
       
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                n_sents, dtype=torch.long, device=top_vecs.device
            )
#            print("########position_ids",position_ids.shape, position_ids)
            position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
#            print("########position_ids",position_ids.shape, position_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
              
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
        
        
        para_position_embeddings = self.position_embeddings(para_pos)
        sent_position_embeddings = self.position_embeddings(sent_pos)
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
        
        embeddings = (           
             position_embeddings          
            + sent_struct_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        self.add_embeddings = embeddings
        return embeddings
    

    
class HiStructBertModel(BertModel):

    config_class = HiStructBertConfig
    pretrained_model_archive_map = HiStructBert_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"
    
#    def __init__(self, large, temp_dir, config):
#        super(HiStructBertModel, self).__init__(config)
#        if(large):
#            self = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
#        else:
#            self = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        #self.finetune = finetune
        
    def __init__(self, config):
        super(HiStructBertModel, self).__init__(config)
        #self.embeddings = HiStructBertModel(config)
        #self.init_weights()

    def forward(
        self,
        input_ids,
        tok_struct_vec,
        sent_struct_vec,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, tok_struct_vec=tok_struct_vec, sent_struct_vec=sent_struct_vec, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


#class LayoutlmForTokenClassification(BertPreTrainedModel):
#    config_class = LayoutlmConfig
#    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
#    base_model_prefix = "bert"
#
#    def __init__(self, config):
#        super().__init__(config)
#        self.num_labels = config.num_labels
#        self.bert = LayoutlmModel(config)
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#
#        self.init_weights()
#
#    def forward(
#        self,
#        input_ids,
#        bbox,
#        attention_mask=None,
#        token_type_ids=None,
#        position_ids=None,
#        head_mask=None,
#        inputs_embeds=None,
#        labels=None,
#    ):
#
#        outputs = self.bert(
#            input_ids=input_ids,
#            bbox=bbox,
#            attention_mask=attention_mask,
#            token_type_ids=token_type_ids,
#            position_ids=position_ids,
#            head_mask=head_mask,
#        )
#
#        sequence_output = outputs[0]
#
#        sequence_output = self.dropout(sequence_output)
#        logits = self.classifier(sequence_output)
#
#        outputs = (logits,) + outputs[
#            2:
#        ]  # add hidden states and attention if they are here
#        if labels is not None:
#            loss_fct = CrossEntropyLoss()
#            # Only keep active parts of the loss
#            if attention_mask is not None:
#                active_loss = attention_mask.view(-1) == 1
#                active_logits = logits.view(-1, self.num_labels)[active_loss]
#                active_labels = labels.view(-1)[active_loss]
#                loss = loss_fct(active_logits, active_labels)
#            else:
#                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#            outputs = (loss,) + outputs
#
#        return outputs  # (loss), scores, (hidden_states), (attentions)
#
#
#class LayoutlmForSequenceClassification(BertPreTrainedModel):
#    config_class = LayoutlmConfig
#    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
#    base_model_prefix = "bert"
#
#    def __init__(self, config):
#        super(LayoutlmForSequenceClassification, self).__init__(config)
#        self.num_labels = config.num_labels
#
#        self.bert = LayoutlmModel(config)
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
#
#        self.init_weights()
#
#    def forward(
#        self,
#        input_ids,
#        bbox,
#        attention_mask=None,
#        token_type_ids=None,
#        position_ids=None,
#        head_mask=None,
#        inputs_embeds=None,
#        labels=None,
#    ):
#
#        outputs = self.bert(
#            input_ids=input_ids,
#            bbox=bbox,
#            attention_mask=attention_mask,
#            token_type_ids=token_type_ids,
#            position_ids=position_ids,
#            head_mask=head_mask,
#        )
#
#        pooled_output = outputs[1]
#
#        pooled_output = self.dropout(pooled_output)
#        logits = self.classifier(pooled_output)
#
#        outputs = (logits,) + outputs[
#            2:
#        ]  # add hidden states and attention if they are here
#
#        if labels is not None:
#            if self.num_labels == 1:
#                #  We are doing regression
#                loss_fct = MSELoss()
#                loss = loss_fct(logits.view(-1), labels.view(-1))
#            else:
#                loss_fct = CrossEntropyLoss()
#                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#            outputs = (loss,) + outputs
#
#        return outputs  # (loss), logits, (hidden_states), (attentions)
