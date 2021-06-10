import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from pytorch_transformers import RobertaModel
from transformers import LongformerModel,LongformerConfig
from transformers import PegasusTokenizer, BigBirdPegasusModel
from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import BigBirdPegasusLearnedPositionalEmbedding
from torch.nn.init import xavier_uniform_
from others.logging import logger,init_logger
from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from models.histruct.histructbert import HiStructBert
from models.histruct.tokStructEmb import LATokInputEmb, LPTokInputEmb,SINTokInputEmb



def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, base_LM, temp_dir, finetune):
        super(Bert, self).__init__()
        if(base_LM=='bert-large'):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
            #self.model = HiStructBertModel(large, temp_dir)
        elif(base_LM=='bert-base'):
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec

class Roberta(nn.Module):
    def __init__(self, base_LM, temp_dir, finetune):
        super(Roberta, self).__init__()
        self.model = RobertaModel.from_pretrained(base_LM, cache_dir=temp_dir)
        
        self.finetune = finetune

    def forward(self, x, mask):
        if(self.finetune):
            top_vec, _ = self.model(x,  attention_mask=mask)
#            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, attention_mask=mask)
#                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec
    
class Longformer(nn.Module):
    def __init__(self, args):
        super(Longformer, self).__init__()
        
        config = LongformerConfig.from_pretrained('allenai/'+args.base_LM) 
        config.attention_window=args.local_attention_window

        self.model = LongformerModel.from_pretrained('allenai/'+args.base_LM, cache_dir=args.temp_dir,config=config)      
        self.finetune = args.finetune_bert
        self.use_global_attention = args.use_global_attention

    def forward(self, x, mask, clss):
        #position_ids
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)     
        position_ids = position_ids.unsqueeze(0).expand_as(x)

        
        #attention_mask
        attention_mask = mask.long()#torch.ones(x.shape, dtype=torch.long, device=x.device)# initialize to local attention (0)
        

        #global_attention_mask    
        global_attention_mask = torch.zeros(x.shape, dtype=torch.long, device=x.device)
        global_attention_mask[:, clss] = 1
#        print('#########global_attention_mask ')
#        print(global_attention_mask)
        
        if(self.finetune):
#            outputs = self.model(x, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
#            top_vec = outputs.last_hidden_state
            #print('######## Finetune')
            if (self.use_global_attention):
                top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask).last_hidden_state
            else:
                top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=None).last_hidden_state
            
        else:
            #print('######## Not Finetune')
            self.eval()
            with torch.no_grad():
#                outputs = self.model(x, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
#                top_vec = outputs.last_hidden_state
                if (self.use_global_attention):
                    top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask).last_hidden_state
                else:
                    top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=None).last_hidden_state
        return top_vec
    
class BigBirdPegasus(nn.Module):
    def __init__(self, args):
        super(BigBirdPegasus, self).__init__()
        
        self.model = BigBirdPegasusModel.from_pretrained('google/'+args.base_LM, cache_dir=args.temp_dir)
        self.finetune = args.finetune_bert

    def forward(self, x, mask):
        
        if(self.finetune):
            
            top_vec  = self.model(x,  attention_mask=mask).last_hidden_state
            
        else:
            
            self.eval()
            with torch.no_grad():
                top_vec  = self.model(x, attention_mask=mask).last_hidden_state
#        print('top_vec',top_vec.shape,top_vec)       
        return top_vec

class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        
        
        init_logger(self.args.log_file)
        
        #print("#####self.bert.model.embeddings",self.bert.model.embeddings)
        
        if (args.add_tok_struct_emb):
            self.bert = HiStructBert(args.base_LM, args.temp_dir, args.finetune_bert)
            logger.info("#####Input embeddings_add token hierarchical structure embeddings: TRUE") 
            if (args.tok_pos_emb_type == 'learned_all'):
                logger.info("-----Type of positional embeddings...learnable")
                logger.info("-----Sequential position and hiarchical positions...different PosEmbs ")
                logger.info("-----Token Structure Embeddings_combination mode ... "+args.tok_se_comb_mode)
                
                self.bert.model.embeddings = LATokInputEmb(self.bert.model.config, args)
               
            elif (args.tok_pos_emb_type == 'learned_pos'):
                logger.info("-----Type of positional embeddings...learnable")
                logger.info("-----Sequential position and hiarchical positions...one same PosEmb ")
                logger.info("-----Token Structure Embeddings_combination mode ... "+args.tok_se_comb_mode)
                
                self.bert.model.embeddings = LPTokInputEmb(self.bert.model.config, args)
               
                    
            elif (args.tok_pos_emb_type == 'sinusoidal'):
                logger.info("-----Type of positional embeddings...sinusoidal")
                logger.info("-----Token Structure Embeddings_combination mode ... "+args.tok_se_comb_mode)
                
                self.bert.model.embeddings = SINTokInputEmb(self.bert.model.config, args)
            else:
                raise ValueError("Must choose one from: ['learned_pos', 'learned_all', 'sinusoidal'] as args.tok_pos_emb_type")
                
                
        else:
            if (args.base_LM.startswith('bert')):
                self.bert = Bert(args.base_LM, args.temp_dir, args.finetune_bert)
            elif (args.base_LM.startswith('roberta')):
                self.bert = Roberta(args.base_LM, args.temp_dir, args.finetune_bert)
            elif (args.base_LM.startswith('longformer')):
                self.bert = Longformer(args)
            elif (args.base_LM.startswith('bigbird-pegasus')):
                self.bert = BigBirdPegasus(args)
                
            logger.info("#####Input embeddings_add token hierarchical structure embeddings: FALSE")
            logger.info("-----use original BERT learnable PosEmb, base LM: "+args.base_LM)
        
                
           
        self.ext_layer = ExtTransformerEncoder(self.bert.model,args, self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.base_LM.startswith('bert') and args.max_pos>512):
#            print('1#####self.bert.model.config.max_position_embeddings',self.bert.model.config.max_position_embeddings)
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
#            print('#####self.bert.model.config.max_position_embeddings',self.bert.model.config.max_position_embeddings)
        
        if(args.base_LM.startswith('longformer') and args.max_pos>4098):
#            print('2#####self.bert.model.config.max_position_embeddings',self.bert.model.config.max_position_embeddings)
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:4098] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[4098:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-4098,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
            
            
        if(args.base_LM.startswith('bigbird-pegasus') and args.max_pos>4096):
#            print('2#####self.bert.model.config.max_position_embeddings',self.bert.model.config.max_position_embeddings)
#            print(type(self.bert.model.encoder.embed_positions),self.bert.model.encoder.embed_positions)
            my_pos_embeddings = BigBirdPegasusLearnedPositionalEmbedding(args.max_pos, self.bert.model.config.hidden_size)
            #my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:4096] = self.bert.model.encoder.embed_positions.weight.data
            my_pos_embeddings.weight.data[4096:] = self.bert.model.encoder.embed_positions.weight.data[-1][None,:].repeat(args.max_pos-4096,1)
            self.bert.model.encoder.embed_positions = my_pos_embeddings
#            print('#####self.bert.model.config.max_position_embeddings',self.bert.model.config.max_position_embeddings)
        
        


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls,sent_struct_vec,tok_struct_vec, section_names):#
        if (self.args.base_LM.startswith('bert')):
            if (self.args.add_tok_struct_emb):
                top_vec = self.bert(src, segs, mask_src, tok_struct_vec)
            else: 
                top_vec = self.bert(src, segs, mask_src)
        else:
            if (self.args.add_tok_struct_emb):
                logger.info('add_tok_struct_emb is not implemented for the base model %s, please set -add_tok_struct_emb false'%self.args.base_LM)
                exit()
            else: 
                if self.args.base_LM.startswith('longformer'):
                    top_vec = self.bert(src, mask_src, clss)
                else:
                    top_vec = self.bert(src, mask_src)
                    
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#        print('sents_vec',sents_vec.shape,sents_vec)
        sents_vec = sents_vec * mask_cls[:, :, None].float()
#        print('sents_vec',sents_vec.shape,sents_vec)
#        sent_scores = self.ext_layer(sents_vec, mask_cls,sent_struct_vec,tok_struct_vec).squeeze(-1)
        sent_scores = self.ext_layer(sents_vec, mask_cls, sent_struct_vec,section_names).squeeze(-1)
        return sent_scores, mask_cls
    
#class ExtSummarizerSent(nn.Module):
#    def __init__(self, args, device, checkpoint):
#        super(ExtSummarizerSent, self).__init__()
#        self.args = args
#        self.device = device
#        
#        
#        init_logger(self.args.log_file)
#        
#        #print("#####self.bert.model.embeddings",self.bert.model.embeddings)
#        
#        
#        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
#        logger.info("#####Input embeddings_add token hierarchical structure embeddings: FALSE")
#        logger.info("-----use original BERT learnable PosEmb") 
#        
#                
#           
#        self.ext_layer = ExtTransformerEncoder(self.bert.model,args, self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
#                                               args.ext_dropout, args.ext_layers)
#        
#        if (args.encoder == 'baseline'):
#            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
#                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
#            self.bert.model = BertModel(bert_config)
#            self.ext_layer = Classifier(self.bert.model.config.hidden_size)
#
#        if(args.max_pos>512):
#            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
#            #my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
#            #my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
#            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
#        
#        
#
#
#        if checkpoint is not None:
#            self.load_state_dict(checkpoint['model'], strict=True)
#        else:
#            if args.param_init != 0.0:
#                for p in self.ext_layer.parameters():
#                    p.data.uniform_(-args.param_init, args.param_init)
#            if args.param_init_glorot:
#                for p in self.ext_layer.parameters():
#                    if p.dim() > 1:
#                        xavier_uniform_(p)
#
#        self.to(device)
#
#    def forward(self, src, segs, clss, mask_src, mask_cls,sent_struct_vec):#
#       
#        top_vec = self.bert(src, segs, mask_src)
#        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#        sents_vec = sents_vec * mask_cls[:, :, None].float()
##        sent_scores = self.ext_layer(sents_vec, mask_cls,sent_struct_vec,tok_struct_vec).squeeze(-1)
#        sent_scores = self.ext_layer(sents_vec, mask_cls, sent_struct_vec).squeeze(-1)
#        return sent_scores, mask_cls
    



#class AbsSummarizer(nn.Module):
#    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
#        super(AbsSummarizer, self).__init__()
#        self.args = args
#        self.device = device
#        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
#
#        if bert_from_extractive is not None:
#            self.bert.model.load_state_dict(
#                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)
#
#        if (args.encoder == 'baseline'):
#            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
#                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
#                                     intermediate_size=args.enc_ff_size,
#                                     hidden_dropout_prob=args.enc_dropout,
#                                     attention_probs_dropout_prob=args.enc_dropout)
#            self.bert.model = BertModel(bert_config)
#
#        if(args.base_LM.startswith('bert') and args.max_pos>512):
#            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
#            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
#            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
#            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
#        self.vocab_size = self.bert.model.config.vocab_size
#        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#        if (self.args.share_emb):
#            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#
#        self.decoder = TransformerDecoder(
#            self.args.dec_layers,
#            self.args.dec_hidden_size, heads=self.args.dec_heads,
#            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
#
#        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
#        self.generator[0].weight = self.decoder.embeddings.weight
#
#
#        if checkpoint is not None:
#            self.load_state_dict(checkpoint['model'], strict=True)
#        else:
#            for module in self.decoder.modules():
#                if isinstance(module, (nn.Linear, nn.Embedding)):
#                    module.weight.data.normal_(mean=0.0, std=0.02)
#                elif isinstance(module, nn.LayerNorm):
#                    module.bias.data.zero_()
#                    module.weight.data.fill_(1.0)
#                if isinstance(module, nn.Linear) and module.bias is not None:
#                    module.bias.data.zero_()
#            for p in self.generator.parameters():
#                if p.dim() > 1:
#                    xavier_uniform_(p)
#                else:
#                    p.data.zero_()
#            if(args.use_bert_emb):
#                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#                self.decoder.embeddings = tgt_embeddings
#                self.generator[0].weight = self.decoder.embeddings.weight
#
#        self.to(device)
#
#    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
#        top_vec = self.bert(src, segs, mask_src)
#        dec_state = self.decoder.init_decoder_state(src, top_vec)
#        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
#        return decoder_outputs, None
