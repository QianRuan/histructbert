import gc
import glob
import hashlib
import itertools
import json
import os
import statistics
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
import shutil

import torch
from multiprocess import Pool

from others.logging import logger,init_logger
from others.tokenization import BertTokenizer
from pytorch_transformers import XLNetTokenizer
#from pytorch_transformers import BertTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET



nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)

def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


    



def obtain_histruct_info(doc, args, tokenizer):
    #list of tokens
    src_sent_tokens = doc['src'] #src_sent_tokens = doc['src_sent_tokens'] 
    src_para_tokens = doc['src_para_tokens']
    
#    idxs = [i for i, s in enumerate(src_sent_tokens) if (len(s) > args.min_src_ntokens_per_sent)]   # 
#    src_sent_tokens = [src_sent_tokens[i][:args.max_src_ntokens_per_sent] for i in idxs]      # 
#    src_sent_tokens = src_sent_tokens[:args.max_src_nsents]#
    
    #list of combined text    
    src_sent = [' '.join(sent) for sent in src_sent_tokens]
    src_para = [' '.join(para) for para in src_para_tokens]

    #list of tokens (tokenized by (bert) tokenizer)  
    src_sent_tokens_bert = [tokenizer.tokenize(sent) for sent in src_sent]  
    src_para_tokens_bert = [tokenizer.tokenize(para) for para in src_para]  
    
    src_sent_tokens_bert_cp = src_sent_tokens_bert.copy()
        
#    print("####################src_sent_tokens",len(src_sent_tokens),src_sent_tokens)
#    print("####################src_para_tokens",len(src_para_tokens),src_para_tokens)
#    print("####################src_sent",len(src_sent),src_sent)
#    print("####################src_para",len(src_para),src_para)
#    print("####################src_sent_tokens_bert",len(src_sent_tokens_bert),src_sent_tokens_bert)
#    print("####################src_para_tokens_bert",len(src_para_tokens_bert),src_para_tokens_bert)
    
    
    
    overall_sent_pos=[ i for i in range(len(src_sent))]
 
    #------------------------------------------------------------------------------------------------obtain sentence structure info
    sent_struct_vec=[]
    for i in range(len(src_para_tokens_bert)):
        #print("###1")
        sent_idx_in_para=0
        for j in range(len(src_sent_tokens_bert)):
            #print("###2")
            #print(source_sent_tokens[j])
            #print(source_para_tokens[i])
            #print("###3")
            
            #print("###4")
            if (src_sent_tokens_bert[j]!=[]) and (src_para_tokens_bert[i][:len(src_sent_tokens_bert[j])] == src_sent_tokens_bert[j]):
                #print("###5")
                sent_struct_vec.append((i,sent_idx_in_para))
                
                src_para_tokens_bert[i] = src_para_tokens_bert[i][len(src_sent_tokens_bert[j]):]
                src_sent_tokens_bert[j] = []
                
                #print(source_sent_tokens)
                #print(source_para_tokens)
                #print(sent_struct_vec)
                if src_para_tokens_bert[i] != []:
                    sent_idx_in_para+=1
                    #print("continue")
                    continue
                if src_para_tokens_bert[i] == []:
                    #print("break")
                    break               

    #------------------------------------------------------------------------------------------------obtain token structure info
    token_struct_vec=[]
#    print('-------------------')

    if (len(src_sent_tokens_bert_cp)!=len(sent_struct_vec)):
            print(len(src_sent_tokens_bert_cp),src_sent_tokens_bert_cp)
            print(len(sent_struct_vec),sent_struct_vec)
            raise ValueError("1###len(src_sent_tokens_bert_cp)!=len(sent_struct_vec)")
            
        
    for i in range(len(src_sent_tokens_bert_cp)):
        sent = src_sent_tokens_bert_cp[i]
        #print("####",i)
        a = sent_struct_vec[i][0]
        b = sent_struct_vec[i][1]
        #struct_vec for [CLS]
        #token_struct_vec.append((a,b,0))
        #print("len(sent)",len(sent))
        #print(i,sent)
        sent_tok_struct_vec=[]
        #struct_vec for [CLS]
        sent_tok_struct_vec.append((a,b,0))
        for j in range(len(sent)):
            #print(j)
            sent_tok_struct_vec.append((a,b,j+1))
        #struct_vec for [SEP]
        sent_tok_struct_vec.append((a,b,j+2))
        token_struct_vec.append(sent_tok_struct_vec)
        
        #token_struct_vec.append((a,b,j+2))
    if (len(token_struct_vec)!=len(src_sent_tokens_bert_cp)):
            print(len(token_struct_vec),token_struct_vec)
            print(len(src_sent_tokens_bert_cp),src_sent_tokens_bert_cp)
            raise ValueError("2###len(token_struct_vec)!=len(src_sent_tokens_bert_cp)")
    
    for i in range(len(token_struct_vec)):
        if len(src_sent_tokens_bert_cp[i])+2 != len(token_struct_vec[i]) :
            print(len(src_sent_tokens_bert_cp[i]),src_sent_tokens_bert_cp[i])
            print(len(token_struct_vec[i]),token_struct_vec[i])
            raise ValueError("3###len(src_sent_tokens_bert_cp[i])+2 != len(token_struct_vec[i])")
                            
    return overall_sent_pos, sent_struct_vec, token_struct_vec


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    #print("######abstract ",abstract)
    abstract = _rouge_clean(' '.join(abstract)).split()
    #print("######abstract ",len(abstract),abstract)
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    #print("######sents ",len(sents),sents)
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    #print("######predicted_1grams, len = ",len(evaluated_1grams),evaluated_1grams)
    reference_1grams = _get_word_ngrams(1, [abstract])
    #print("######gold_1grams, len =  ",len(reference_1grams),reference_1grams)
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    #print("######predicted_2grams, len =   ",len(evaluated_2grams),evaluated_2grams)
    reference_2grams = _get_word_ngrams(2, [abstract])
    #print("######gold_2grams, len =   ", len(reference_2grams),reference_2grams)

    selected = []
    #print("######summary_size ", summary_size)
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
#        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=True)


        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, doc, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):
        
        init_logger(self.args.log_file)
        
        if ((not is_test) and len(src) == 0):
            logger.info('######---- not preprocessed')
            return None
        #--------------------------------------------------------------------------------------------------------histruct info
        overall_sent_pos, _sent_struct_vec, _token_struct_vec = obtain_histruct_info(doc,self.args,self.tokenizer)
        
              
        #--------------------------------------------------------------------------------------------------------min/max_src_ntokens_per_sent
        #print("#################src",len(src),src)
        original_src_txt = [' '.join(s) for s in src]
        #print("#################original_src_txt",len(original_src_txt),original_src_txt)

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        #print("#################self.args.min_src_ntokens_per_sent",self.args.min_src_ntokens_per_sent)
        #print("#################idxs",len(idxs),idxs)
        
        
        #--------------------------------------------------------------------------------------------------------sent_labels
        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1
#        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
#        token_struct_vec = [_token_struct_vec[i][:self.args.max_src_ntokens_per_sent+2] for i in idxs]##
        
        src = [src[i] for i in idxs]
        token_struct_vec = [_token_struct_vec[i] for i in idxs]##
        #print("#################self.args.max_src_ntokens_per_sent",self.args.max_src_ntokens_per_sent)
        #print("#################idxs",len(src),src)
        #remove short sentences
        sent_labels = [_sent_labels[i] for i in idxs]
        sent_struct_vec = [_sent_struct_vec[i] for i in idxs]#######
        #print("#################sent_labels",len(sent_labels),sent_labels)
        
#        src = src[:self.args.max_src_nsents]
#        token_struct_vec = token_struct_vec[:self.args.max_src_nsents]
        
        token_struct_vec = sum(token_struct_vec,[]) #flat list
        #print("#################idxs",len(src),src)
        
#        sent_labels = sent_labels[:self.args.max_src_nsents]
#        sent_struct_vec = sent_struct_vec[:self.args.max_src_nsents]######
        
        #print("#################self.args.max_src_nsents",self.args.max_src_nsents)
        #print("#################sent_labels",len(sent_labels),sent_labels)
        
        
        #--------------------------------------------------------------------------------------------------------min_src_nsents
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None
        

        #--------------------------------------------------------------------------------------------------------src_subtoken_idxs
        src_txt = [' '.join(sent) for sent in src]#
        #print("#################src_txt",len(src_txt),src_txt)
        #add [CLS] and [SEP]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)#
        #print("#################text",len(text), text)
        src_subtokens = self.tokenizer.tokenize(text)#
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        #print("#################src_subtokens",len(src_subtokens), src_subtokens)
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        #print("#################src_subtokens",len( src_subtoken_idxs),  src_subtoken_idxs)
       
        
        
        #--------------------------------------------------------------------------------------------------------segments_ids
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]    
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
       
                
        #--------------------------------------------------------------------------------------------------------cls_ids      
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid] #index of [CLS]
        #print("#################cls_ids",len(cls_ids), cls_ids)
        
        
        #--------------------------------------------------------------------------------------------------------sent_labels
        sent_labels = sent_labels[:len(cls_ids)]
        
        #--------------------------------------------------------------------------------------------------------tgt_subtoken_idxs
        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()
#         tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)        
        
        #--------------------------------------------------------------------------------------------------------src_txt, tgt_txt
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        
        
        
        #check length
        flag=False
        if not (len(sent_labels)==len(cls_ids)==len(sent_struct_vec)): #src_txt still contains short sentences, ignore
            print('1----Length should be the same')
            print(len(sent_labels))
            print(len(cls_ids))
#            print(len(src_txt),src_txt)#
            print(len(sent_struct_vec))
            
            
           
        if not (len(segments_ids)==len(src_subtoken_idxs)==len(token_struct_vec)):
            flag=True
            print('2----Length should be the same')
            print(len(segments_ids))#,segments_ids)
            print(len(src_subtoken_idxs))#,src_subtoken_idxs)
#            print(len(src_txt),src_txt)#
            print(len(token_struct_vec))#,token_struct_vec)
            
       
            
#            
#            if not (len(segments_ids)==len(src_subtoken_idxs)):
#                raise ValueError('2----Length should be the same')
#            if not (len(token_struct_vec)==len(src_subtoken_idxs)):
#                raise ValueError('2----Length should be the same')
            
                
            
            #raise ValueError('2----Length should be the same')
            
        #subtokens to sentences 
        li=[]
        lists=[]
        for idx in src_subtoken_idxs:
            if not idx==102:
                li.append(idx)
            else:
                li.append(idx)
                lists.append(li)
                li=[]
                continue
        #print(lists)
        
        #token_struct_vec to sentences
        li2=[]
        lists2=[]
        count=0
        t=token_struct_vec
        for i in range(len(t)):
           
            if t[i][2]==0:                
                count+=1
                if count==1 :
                    li2.append(t[i])
                elif count>1:
                    lists2.append(li2)
                    li2=[]
                    li2.append(t[i])
                    count=1 
            else:
                li2.append(t[i])
                if i == len(t)-1:
                    lists2.append(li2)
                    
        #segment_ids_vec to sentences
        li3=[]
        lists3=[]
        for i in range(len(segments_ids)):
            if i==0:
                li3.append(segments_ids[i])            
            elif segments_ids[i-1]!=segments_ids[i]:
                lists3.append(li3)
                li3=[]
                li3.append(segments_ids[i])
            else:
                li3.append(segments_ids[i])
                if i==len(segments_ids)-1:
                    lists3.append(li3)
                    
                    
        
                    
        #both contains same number of sentences 
        
        if (len(lists)!=len(lists2)!=len(lists3)):
            flag=True
            print('3----Nr. of sentences should be the same')
            
            
        #sentences at same index contain same number of items (sutokens and its structure vector)          
        l=[len(x) for x in lists]
        l2=[len(x) for x in lists2]
        l3=[len(x) for x in lists3]

        for i in range(len(l)):
            if l[i]!=l2[i]:
                flag=True
                print("4----Nr. of items in the same sentence should be the same: Subtoken_idxs vs. token_struct_vec")
                print(i, l[i], l2[i])
                print (lists[i])
                print (lists2[i])
#                    print(src_txt[i])   
            if l[i]!=l3[i]:
                flag=True
                print("5----Nr. of items in the same sentence should be the same: Subtoken_idxs vs. segment_ids")
                print(i, l[i], l3[i])
                print (lists[i])
                print (lists3[i])
#                    print(src_txt[i])   
            if l2[i]!=l3[i]:
                flag=True
                print("6----Nr. of items in the same sentence should be the same: token_struct_vec vs. segment_ids")
                print(i, l2[i], l3[i])
                print (lists2[i])
                print (lists3[i])
        
        if (flag) :
            raise ValueError("7----length check failed")
            

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, overall_sent_pos, sent_struct_vec, token_struct_vec


def format_to_histructbert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):       
            #real_name = json_f.split('/')[-1]
            real_name = json_f.split('/')[-1].split('\\')[-1]
            #print("##########################real_name",real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        #print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_histructbert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_histructbert(params):
   
    
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    init_logger(args.log_file)
    
    #check if the save file already exists
    if (os.path.exists(save_file)):
        logger.info("Save file %s already exisits, remove it!" %(save_file))
        os.remove(save_file)

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file, encoding='utf-8')) #nr. of documents
    #print("####################jobs",len(jobs))
    datasets = []
#    logger.info("Do greedy selection to create oracle summaries, summary size:"+str(args.summ_size))
    logger.info("Do greedy selection to create oracle summaries, summary size: long")
    for d in jobs:
        #get list of source sentences and gold summary sentences
        source, tgt = d['src'], d['tgt']
        
        #get index of selected sentences (in oracle summary)
        
        sent_labels = greedy_selection(source, tgt, len(source)) #!!!!len(source)
#         sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, args.summ_size) #!!!!len(source)
        tgt_sent_idx = sent_labels
        #print("####################sent_labels",len(sent_labels),sent_labels)#可能有2
        
        #do lowercase
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
            #print("####################sourcelower",len(source),source)
            #print("####################tgtlower",len(tgt),tgt)
                  
        b_data = bert.preprocess(d, source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)

        if (b_data is None):
            continue
        
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt,overall_sent_pos, sent_struct_vec, token_struct_vec = b_data
        
              
        b_data_dict = {"src": src_subtoken_idxs, ##!!
                       "tgt": tgt_subtoken_idxs, #!!
                       "src_sent_labels": sent_labels, ##!!
                       "segs": segments_ids, ##!!
                       'clss': cls_ids, ##!!
                       'src_txt': src_txt, ##!!
                       "tgt_txt": tgt_txt, ##!!
#                       "tgt_sent": tgt, #-- 
#                       "tgt_sent_idx":tgt_sent_idx,#--
#                       "overall_sent_pos":overall_sent_pos,#--
                       "sent_struct_vec":sent_struct_vec, ##!!
                       "token_struct_vec":token_struct_vec}##!!
        
#        if (len(b_data_dict['src_sent_labels'])==len(b_data_dict['clss'])==len(b_data_dict['src_txt'])==len(b_data_dict['sent_struct_vec'])):
#             if (len(b_data_dict['segs'])==len(b_data_dict['src'])==len(b_data_dict['token_struct_vec'])):
#                 datasets.append(b_data_dict)
             

        datasets.append(b_data_dict)
#        #len = nr. of sentences
#        print("#################src_sent_labels",len(b_data_dict['src_sent_labels']),b_data_dict['src_sent_labels']) 
#        print("#################clss",len(b_data_dict['clss']),b_data_dict['clss'])
#        print("#################src_txt",len(b_data_dict['src_txt']),b_data_dict['src_txt'])
#        print("#################overall_sent_pos",len(b_data_dict['overall_sent_pos']),b_data_dict['overall_sent_pos'])
#        print("#################sent_struct_vec, len =",len(b_data_dict['sent_struct_vec']),b_data_dict['sent_struct_vec'])
#        
#        #len = nr. of tokens including [CLS]  and [SEP]   
#        print("#################segs",len(b_data_dict['segs']),b_data_dict['segs'])
#        print("#################src",len(b_data_dict['src']),b_data_dict['src'])
#        print("#################token_struct_vec, len =",len(b_data_dict['token_struct_vec']),b_data_dict['token_struct_vec'])
#        
#        #summary info      
#        print("#################tgt_txt",len(b_data_dict['tgt_txt']),b_data_dict['tgt_txt'])
#        print("#################tgt_sent_idx",len(b_data_dict['tgt_sent_idx']),b_data_dict['tgt_sent_idx'])
#        print("#################tgt",len(b_data_dict['tgt']),b_data_dict['tgt'])
              
        #len = nr. of sentences
#        logger.info("#############################") 
#        logger.info("src_sent_labels "+str(len(b_data_dict['src_sent_labels']))) 
#        logger.info("clss "+str(len(b_data_dict['clss'])))
#        logger.info("src_txt "+str(len(b_data_dict['src_txt'])))
#        #logger.info("overall_sent_pos "+str(len(b_data_dict['overall_sent_pos'])))
#        logger.info("sent_struct_vec "+str(len(b_data_dict['sent_struct_vec'])))
#        logger.info("-------------------") 

#        #len = nr. of tokens including [CLS]  and [SEP]   
#        logger.info("segs "+str(len(b_data_dict['segs'])))
#        logger.info("src "+str(len(b_data_dict['src'])))
#        logger.info("token_struct_vec "+str(len(b_data_dict['token_struct_vec'])))
#        logger.info("-------------------") 
        
        
#         
#        if not (len(b_data_dict['segs'])==len(b_data_dict['src'])==len(b_data_dict['token_struct_vec'])):
#            
#            li=[]
#            lists=[]
#            for idx in b_data_dict['src']:
#                if not idx==102:
#                    li.append(idx)
#                else:
#                    li.append(idx)
#                    lists.append(li)
#                    li=[]
#                    continue
#            #print(lists)
#            
#            li2=[]
#            lists2=[]
#            count=0
#            t=b_data_dict['token_struct_vec']
#            for i in range(len(t)):
#               
#                if t[i][2]==0:                
#                    count+=1
#                    if count==1 :
#                        li2.append(t[i])
#                    elif count>1:
#                        lists2.append(li2)
#                        li2=[]
#                        li2.append(t[i])
#                        count=1
#                        
#                    
#                else:
#                    li2.append(t[i])
#                    if i == len(t)-1:
#                        lists2.append(li2)
#                        
#                    
#            #print(lists2)
#                
#            l=[len(x) for x in lists]
#            l2=[len(x) for x in lists2]
#            print(len(l),l)
#            print(len(l2),l2)
#            
#            for i in range(len(l)):
#                if l[i]!=l2[i]:
#                    print(i, l[i], l2[i])
#                    print (lists[i])
#                    print (lists2[i])
#                    print(b_data_dict['src_txt'][i])
##            
#            raise ValueError('2++++Length should be the same')
#        #summary info 
#        #logger.info("tgt_sent "+str(len(b_data_dict['tgt_sent'])))      
#        #logger.info("tgt_sent_idx "+str(len(b_data_dict['tgt_sent_idx']))+"-----"+" ,".join(str(x) for x in b_data_dict['tgt_sent_idx']))
#        logger.info("tgt_txt "+str(len(b_data_dict['tgt_txt'])))
#        logger.info("tgt "+str(len(b_data_dict['tgt'])))
#        logger.info("#############################") 
              
               
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def merge_data_splits(args):
    corpus_mapping = {}
    init_logger(args.log_file)
    
    save_path = '/'.join(args.save_path.split('/')[:-1])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #split data 
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    logger.info('Mapping documents to train/valid/test datasets')    
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('\\')[1].split('.')[0]#
        #print('##########################')
        #print(real_name)
        #print(corpus_mapping['valid'])
        if (real_name in corpus_mapping['valid']):
            #logger.info(real_name,'in valid')
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            #logger.info(real_name,'in test')
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            #logger.info(real_name,'in train')
            train_files.append(f)
        # else:
        #     train_files.append(f)
    logger.info('There are %s / %s / %s documents in train/valid/test datasets.'% (len(train_files),len(valid_files),len(test_files))) 
    
    ##################################################################################save_statistics
    stat={'nr. docs':(len(train_files),len(valid_files),len(test_files))}
    stat_path = args.save_path.split('/')[0]+'/statistics.json'
    with open(stat_path, 'w+') as save:
                save.write(json.dumps(stat))
    ##################################################################################save_statistics
    
    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}  
    
    logger.info('Merging documents...')   
    #merge data splits
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in a_lst: #pool.imap_unordered(_merge_data_splits, a_lst):#
            #print(d[0])
            doc=json.load(open(d[0], encoding='utf-8'))
            dataset.append(doc)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w+') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            
            save_path = args.save_path
            save_path = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            with open(pt_file, 'w+') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []
    logger.info('DONE')   

def compute_statistics(args):
    
    stat_path = args.histruct_path.split('/')[0]+'/statistics.json'
   
    init_logger(args.log_file)
    logger.info("Computing statistics...")
    
    stat = json.load(open(stat_path, encoding='utf-8'))
    print(stat)
    
    doc_len_para=[]
    doc_len_sent=[]
    doc_len_word=[]  
    summ_len_sent=[]
    summ_len_word=[]
    novel_2grams=[]
    novel_1grams=[]
    
    for f in glob.glob(pjoin(args.histruct_path, '*.json')):
        doc=json.load(open(f, encoding='utf-8'))
        
        doc_len_para.append(len(doc["src_para_tokens"]))
        doc_len_sent.append(len(doc["src"]))
        summ_len_sent.append(len(doc["tgt"]))
        
        flat_src = sum(doc["src"],[])
        flat_summ = sum(doc["tgt"],[])
        doc_len_word.append(len(flat_src)) 
        summ_len_word.append(len(flat_summ))
              
        novel_2grams.append(get_novel_ngrams_percentage(flat_src, flat_summ, 2))
        novel_1grams.append(get_novel_ngrams_percentage(flat_src, flat_summ, 1))

        
    stat = {'avg. doc length(words)': round(statistics.mean(doc_len_word),2), 
           'avg. doc length(sentences)': round(statistics.mean(doc_len_sent),2), 
           'avg. doc length(paragraphs)':round(statistics.mean(doc_len_para),2), 
           'avg. summary length(words)': round(statistics.mean(summ_len_word),2), 
           'avg. summary length(sentences)': round(statistics.mean(summ_len_sent),2), 
           '% novel 1grams in gold summary': round(statistics.mean(novel_1grams),2),
           '% novel 2grams in gold summary': round(statistics.mean(novel_2grams),2)}
    
    logger.info(stat)

    with open(stat_path, 'w+') as save:
        save.write(json.dumps(stat))
    logger.info("DONE")
    
def get_novel_ngrams_percentage(flat_src, flat_summ, ngrams):
    
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)
    
    flat_summ = _rouge_clean(' '.join(flat_summ)).split()
    flat_src = _rouge_clean(' '.join(flat_src)).split()
    summ_ngrams = _get_word_ngrams(ngrams, [flat_summ])
    src_ngrams = _get_word_ngrams(ngrams, [flat_src])
    
    same = len(set(summ_ngrams).intersection(set(src_ngrams)))
    novel = len(summ_ngrams)-same
   
    return round((novel/len(summ_ngrams))*100,2)

def extract_histruct_items(args):
    #tok_sent_dir = os.path.abspath(args.tok_sent_path)
    tok_para_dir = os.path.abspath(args.tok_para_path)
    histruct_dir = os.path.abspath(args.histruct_path)
    
    if not os.path.exists(histruct_dir):
        os.makedirs(histruct_dir)
    init_logger(args.log_file)
    logger.info("Extracting histruct items...")
    
    for f in glob.glob(pjoin(args.tok_sent_path, '*.json')):
       
        real_name = f.split('/')[-1].split('\\')[-1]
        histruct_story_path = histruct_dir+'\\'+real_name
        
        
        source, tgt = obtain_source_summ(f, args.lower)
        src_para_tokens = obtain_para(f,args.lower,tok_para_dir,real_name)
        doc = {'src': source, 'tgt':tgt, 'src_para_tokens':src_para_tokens}

        with open(histruct_story_path, 'w+') as save:
                save.write(json.dumps(doc))
    logger.info("DONE")
        
    
    

#obtain list of sentences and list of paragraphs from source text
#the results are kept to obtain hierarchical positions
def obtain_para(f,lower,tok_para_dir,real_name):
    
    para_story_path = tok_para_dir+'\\'+real_name #same name
    #print("###############sent_story_path",sent_story_path)
    #print("###############para_story_path",para_story_path)
    
    #get list of tokens of sentences from source text
#    source_sent_tokens=[]
#    flag =False
#    for sent in json.load(open(sent_story_path, encoding='utf-8'))['sentences']:
#        tokens = [t['word'] for t in sent['tokens']]
#        if (lower):
#            tokens = [t.lower() for t in tokens]
#        if(tokens[0] == '@highlight'):
#            flag = True
#            #print("tokens",tokens)
#            #print("flag",flag)
#            continue
#        if (not flag):
#            source_sent_tokens.append(tokens)
    #print (len(source_sent_tokens),source_sent_tokens)
    
    #get list of tokens of paragraphs from source text
    source_para_tokens=[]
    tgt_para_tokens=[]
    flag =False
    for sent in json.load(open(para_story_path, encoding='utf-8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if(tokens[0] == '@highlight'):
            flag = True
            tgt_para_tokens.append([])
            continue
        if (flag):
            tgt_para_tokens[-1].extend(tokens)
        else:
            source_para_tokens.append(tokens)
            
    source_para_tokens = [clean(' '.join(para)).split() for para in source_para_tokens]        
    
    return source_para_tokens

def obtain_source_summ(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p, encoding='utf-8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tok_sent_dir = os.path.abspath(args.tok_sent_path)
    tok_para_dir = os.path.abspath(args.tok_para_path)
    init_logger(args.log_file)
    
    
    if os.path.exists(tok_sent_dir):
        logger.info("SENT - Save path %s already exisits, remove it!" % (tok_sent_dir))
        shutil.rmtree(tok_sent_dir)
    if os.path.exists(tok_para_dir):
        logger.info("PARA - Save path %s already exisits, remove it!" % (tok_para_dir))
        shutil.rmtree(tok_para_dir)
        
        
        
    logger.info("Preparing to tokenize %s" % (stories_dir))    
    stories = os.listdir(stories_dir)
    # make IO list file
    logger.info("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))

#    #split sentences
#    command_sent = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
#               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
#               'json', '-outputDirectory', tok_sent_dir]
#    
#    #split paragraphs
#    command_para = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
#               '-ssplit.eolonly', 'true', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
#               'json', '-outputDirectory', tok_para_dir]#
    
    #split sentences
    command_sent = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tok_sent_dir]
    
    #split paragraphs
    command_para = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.eolonly', 'true', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tok_para_dir]#
    
    logger.info("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_sent_dir))
    subprocess.call(command_sent,shell=True)
    logger.info("SENT - Stanford CoreNLP Tokenizer has finished.")
    #os.remove("mapping_for_corenlp.txt")
    
    logger.info("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_para_dir))
    subprocess.call(command_para,shell=True)#
    logger.info("PARA - Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized_para = len(os.listdir(tok_para_dir))
    num_tokenized_sent = len(os.listdir(tok_sent_dir))
    if num_orig != num_tokenized_sent:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tok_sent_dir, num_tokenized_sent, stories_dir, num_orig))
    if num_orig != num_tokenized_para:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tok_para_dir, num_tokenized_para, stories_dir, num_orig))
    logger.info("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_para_dir))
    logger.info("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_sent_dir))