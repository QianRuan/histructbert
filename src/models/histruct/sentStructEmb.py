from others.logging import logger,init_logger
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import BertConfig, BertModel, BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertLayerNorm


def compute_se(pos, pe, position_embeddings): 

    x_position_embeddings = torch.zeros_like(position_embeddings)
 
    for i in range(pos.size(0)):
  
        for j in range (pos.size(1)):
        
            idx = int(pos[i][j].item())
            x_position_embeddings[i][j] = pe[0][idx]
            

    return x_position_embeddings



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
        position_ids=None,
    ):
#        print("####inputs",inputs.shape)
        batch_size = inputs.size(0)
        n = inputs.size(1)
#        print("####pe1",self.pe.shape)
        pe = self.pe[:, :n]
#        print("####pe2",pe.shape)
        
        pos_embs = pe.expand(batch_size,-1,-1)
#        print("####pos",pos_embs.shape)
               
        return pe, pos_embs

class LASentAddEmb(nn.Module):
    
    def __init__(self, args, config):
       
        super(LASentAddEmb, self).__init__()
        
        self.args = args
        self.position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
        
        
        
        if(self.args.sent_se_comb_mode == 'concat'):
            if args.max_npara==0:
                self.a_position_embeddings = nn.Embedding(args.max_nsent, int(config.hidden_size/2))
            else:
                self.a_position_embeddings = nn.Embedding(args.max_npara, int(config.hidden_size/2))
            if args.max_nsent_in_para==0:
                self.b_position_embeddings = nn.Embedding(args.max_nsent, int(config.hidden_size/2))
            else:
                self.b_position_embeddings = nn.Embedding(args.max_nsent_in_para, int(config.hidden_size/2))
                
        else:
            if args.max_npara==0:
                self.a_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
            else:
                self.a_position_embeddings = nn.Embedding(args.max_npara, config.hidden_size)
            if args.max_nsent_in_para==0:
                self.b_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
            else:
                self.b_position_embeddings = nn.Embedding(args.max_nsent_in_para, config.hidden_size)
                
#            self.a_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
#            self.b_position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
       
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        print(config)
        print(config.hidden_size)
        if args.base_LM.startswith('bigbird-pegasus'):
            self.LayerNorm = nn.LayerNorm(config.d_model)
            self.dropout = nn.Dropout(config.dropout)
        elif args.base_LM.startswith('bert'):
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
#        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(n_sents, dtype=torch.long, device=top_vecs.device)
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
        
        if(self.args.sent_se_comb_mode == 'sum'):
            sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
            
        elif(self.args.sent_se_comb_mode == 'mean'):
            sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
            
        elif(self.args.sent_se_comb_mode == 'concat'):
            sent_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings,),2)
            
        else:
            raise ValueError ("args.sent_se_comb_mode must be one of ['sum', 'mean', 'concat']")
            
       
        
        if self.args.without_sent_pos and self.args.para_only:
            
            embeddings = para_position_embeddings
            
        elif self.args.without_sent_pos:
            
            embeddings = sent_struct_embeddings
            
        elif self.args.para_only:
            
            embeddings = (           
                 position_embeddings          
                + para_position_embeddings
            )     
        else:
            
            embeddings = (           
                 position_embeddings          
                + sent_struct_embeddings
            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
class SINSentAddEmb(nn.Module):
    def __init__(self, args, config):
       
        super(SINSentAddEmb, self).__init__()
        
        self.args = args
       
        self.position_embeddings = SinPositionalEncoding(config.hidden_size, max_len=args.max_nsent)
        
        if(self.args.sent_se_comb_mode == 'concat'):
            self.histruct_position_embeddings = SinPositionalEncoding(int(config.hidden_size/2),max_len=args.max_nsent)
        else:
            self.histruct_position_embeddings = None
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
#        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
      
        #768 dim
#        pe, position_embeddings= self.position_embeddings(top_vecs,tok_struct_vec,sent_struct_vec)
        
        batch_size = top_vecs.size(0)
        n = top_vecs.size(1)
        pe = self.position_embeddings.pe[:, :n]      
        position_embeddings = pe.expand(batch_size,-1,-1)
       

        
        #256 dim
        if (self.histruct_position_embeddings != None):
#            hs_pe,hs_position_embeddings = self.histruct_position_embeddings(top_vecs,tok_struct_vec,sent_struct_vec)
            hs_pe = self.histruct_position_embeddings.pe[:, :n]   
            hs_position_embeddings = hs_pe.expand(batch_size,-1,-1)
            
        else:
            hs_pe,hs_position_embeddings = None, None
        
#        print("########position_ids",position_ids.shape, position_ids)
#        print("########input_ids",input_ids.shape, input_ids)
#        print("########words_embeddings",words_embeddings.shape, words_embeddings)
#        print("########position_embeddings",position_embeddings.shape, position_embeddings)
#        print("########tok_struct_vec",tok_struct_vec.shape, tok_struct_vec)
#        print("########sent_struct_vec",sent_struct_vec.shape, sent_struct_vec)
#        print("###sent_struct_vec",sent_struct_vec.shape)      
        para_pos = sent_struct_vec[:,:,0]
        sent_pos = sent_struct_vec[:,:,1]
#        print("###para_pos",para_pos.shape) 
        
      
                    
        if(self.args.sent_se_comb_mode == 'concat'):
            para_position_embeddings = compute_se(para_pos, self.histruct_position_embeddings.pe, hs_position_embeddings)
            sent_position_embeddings = compute_se(sent_pos, self.histruct_position_embeddings.pe, hs_position_embeddings)
        else:
            para_position_embeddings = compute_se(para_pos, self.position_embeddings.pe, position_embeddings)
            sent_position_embeddings = compute_se(sent_pos, self.position_embeddings.pe, position_embeddings)
            
       
        
#        print("########para_pos",para_pos.shape,para_pos)
#        print("########sent_pos",sent_pos.shape,sent_pos)
#        print("########tok_pos",tok_pos.shape,tok_pos)
#        print("########para_position_embeddings",para_position_embeddings.shape,para_position_embeddings)
#        print("########sent_position_embeddings",sent_position_embeddings)
#        print("########tok_position_embeddings",tok_position_embeddings)
        #SUM
        if(self.args.sent_se_comb_mode == 'sum'):
            sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
            
        elif(self.args.sent_se_comb_mode == 'mean'):
            sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
            
        elif(self.args.sent_se_comb_mode == 'concat'):
            sent_struct_embeddings = torch.cat((para_position_embeddings,sent_position_embeddings),2)
            
        else:
            raise ValueError("args.sent_se_comb_mode must be one of ['sum','mean','concat']")
            
        
        if self.args.without_sent_pos and self.args.para_only:
            
            embeddings = para_position_embeddings
            
        elif self.args.without_sent_pos:
            
            embeddings = sent_struct_embeddings
            
        elif self.args.para_only:
           
            embeddings = (           
                 position_embeddings          
                + para_position_embeddings
            )     
        else:
            
            embeddings = (           
                 position_embeddings          
                + sent_struct_embeddings
            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings




    
class LPSentAddEmb(nn.Module):
    def __init__(self, args, config):
       
        super(LPSentAddEmb, self).__init__()
        
        self.args =args
        self.position_embeddings = nn.Embedding(args.max_nsent, config.hidden_size)
        
        if(self.args.sent_se_comb_mode == 'concat'):
#           self.histruct_position_embeddings = nn.Embedding(config.max_position_embeddings, int(config.hidden_size)/2)
            raise ValueError ("Concat mode can not be used when we only learn one PosEmb for all positions")
            
       
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        top_vecs,
#        tok_struct_vec,
        sent_struct_vec,
        position_ids=None,
    ):
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        #seq_length = top_vecs.size(1)
        if position_ids is None:
            position_ids = torch.arange(n_sents, dtype=torch.long, device=top_vecs.device)
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
        
        if(self.args.sent_se_comb_mode == 'sum'):
            sent_struct_embeddings = para_position_embeddings+sent_position_embeddings
            
        elif(self.args.sent_se_comb_mode == 'mean'):
            sent_struct_embeddings = (para_position_embeddings+sent_position_embeddings)/2
            
        else:
            raise ValueError ("args.sent_se_comb_mode must be one of ['sum', 'mean']")
            
        
        if self.args.without_sent_pos and self.args.para_only:
            
            embeddings = para_position_embeddings
            
        elif self.args.without_sent_pos:
            
            embeddings = sent_struct_embeddings
            
        elif self.args.para_only:
            
            embeddings = (           
                 position_embeddings          
                + para_position_embeddings
            )     
        else:
            
            embeddings = (           
                 position_embeddings          
                + sent_struct_embeddings
            )
        
        
#        if self.args.without_sent_pos :
#            embeddings = sent_struct_embeddings
#        else:
#            embeddings = (           
#                 position_embeddings          
#                + sent_struct_embeddings
#            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    
class LPSentAddEmbPOS(nn.Module):
     def __init__(self, args, config):
       
        super(LPSentAddEmbPOS, self).__init__()
    
        self.position_embeddings = nn.Embedding(
            args.max_nsent, config.hidden_size
        )
              
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

     def forward(
        self,
        top_vecs,
#        tok_struct_vec,
#        sent_struct_vec,
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
    


        



    

