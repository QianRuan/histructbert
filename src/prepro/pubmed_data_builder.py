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
from copy import deepcopy

import torch
from multiprocess import Pool

from others.logging import logger,init_logger
from others.tokenization import BertTokenizer
from pytorch_transformers import RobertaTokenizer,RobertaModel
from transformers import PegasusTokenizer

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

    #read and clean data 
    #get list of sentences in article (without gold summary)
    src_sent = doc['article_text'] 
    src_sent=[sent.strip().lower() for sent in src_sent]
    
    #get list of sections which contain lists of sentences in the section
    src_para_sent = doc['sections']
    src_para_sent = [[sent.strip().lower() for sent in para] for para in src_para_sent]

    #get list of section names
    section_names = doc['section_names']
    section_names = [re.sub(r'[^a-zA-Z ]', '', s).lower().strip() for s in section_names]
    assert len(src_para_sent)==len(section_names)
    
    #remove empty sections 
    idxs = [i for i, para in enumerate(src_para_sent) if para!=[''] and para!=[]]
    src_para_sent = [src_para_sent[i] for i in idxs]
    
    #remove corresponding section names and check
    section_names  = [section_names[i] for i in idxs]
    assert len(src_para_sent)==len(section_names)
    
    #remove sentences that are not in the 'article text' but in the 'sections'
    src_para_sent = [[sent for sent in para if sent in src_sent] for para in src_para_sent]
    
    #remove repeated extra sentences in 'sections' (list of paragraphs containing lists of sentences in the paragraph)
    src_sent_cp = src_sent.copy()
    src_para_sent_cp =deepcopy(src_para_sent)#copy nested list
#    src_para_sent_cp2 =deepcopy(src_para_sent)
    sent_in_para_kept = []         
    for h in range(len(src_sent)):
        for i in range(len(src_para_sent)):#nr.of para
            for j in range(len(src_para_sent[i])):
                if src_sent_cp[h]==src_para_sent_cp[i][j]!=None:
                    if sent_in_para_kept!=[] :
                        last_id = sent_in_para_kept[-1]
                        if (last_id[0]==i and last_id[1]<j) or last_id[0]<i:#check location
                            sent_in_para_kept.append((i,j))
                            src_sent_cp[h]=None
                            src_para_sent_cp[i][j]=None
                            break
                        else:
                            continue
                    else:
                        sent_in_para_kept.append((i,j)) 
                        src_sent_cp[h]=None
                        src_para_sent_cp[i][j]=None
                        break
    for i in range(len(src_para_sent)):#nr.of para
            for j in range(len(src_para_sent[i])):
                if (i,j) not in sent_in_para_kept:
                    src_para_sent[i][j]=None
    src_para_sent=[[sent for sent in para if sent!=None] for para in src_para_sent]
                    
    assert len(src_sent)==len(sum(src_para_sent,[]))
    assert src_sent==sum(src_para_sent,[])
#    if src_sent!=sum(src_para_sent,[]):
#        print(len(src_sent),src_sent)
#        print('AFTER',len(src_para_sent), len(sum(src_para_sent,[])), src_para_sent)
#        print('BEFORE',len(src_para_sent_cp2), len(sum(src_para_sent_cp2,[])), src_para_sent_cp2)
#        print(sent_in_para_kept)
#        assert src_sent==sum(src_para_sent,[])
    


    #obtain sentence structure vectors
    para_length = [len(para) for para in src_para_sent]
    sent_struct_vec = []
    for i in range(len(src_para_sent)):
        for j in range(para_length[i]):
            sent_struct_vec.append((i,j))
            
    assert len(sent_struct_vec)==len(src_sent)

    overall_sent_pos = [i for i in range(len(src_sent))]
    skip=False
    skip_reason=''
    
    
                
    #check
    
    #the SE should not be empty
    if sent_struct_vec==[]:
        logger.info('Skipped since the sentence structure vector is empty')
        skip=True
        skip_reason='empty sentence structure vector '
        
    if not args.obtain_tok_se:
        token_struct_vec=None
    else:
        src_sent_tokens_retokenized = [tokenizer.tokenize(sent) for sent in src_sent]  
        token_struct_vec=[]
        
        if not skip:
            
            for i in range(len(src_sent_tokens_retokenized)):
                #current sentence
                sent = src_sent_tokens_retokenized[i]
                #paragraph & sentence positions are same
                a = sent_struct_vec[i][0]
                b = sent_struct_vec[i][1]
                #token structure vectors for current sentence
                sent_tok_struct_vec=[]
                #append struct_vec for [CLS] at begining
                sent_tok_struct_vec.append((a,b,0))
                for j in range(len(sent)):
                    sent_tok_struct_vec.append((a,b,j+1))
                #append struct_vec for [SEP] at end
                sent_tok_struct_vec.append((a,b,j+2))
                
                token_struct_vec.append(sent_tok_struct_vec)
            
            #check
            assert (len(token_struct_vec)==len(src_sent_tokens_retokenized))
            for i in range(len(token_struct_vec)):
                assert (len(src_sent_tokens_retokenized[i])+2 == len(token_struct_vec[i]))
    
        
#        #turncate long sentences
#        if args.max_src_ntokens_per_sent!=0:
#            
#            for i in range(len(token_struct_vec)):        
#                src_sent_list = src_sent_tokens[i][:args.max_src_ntokens_per_sent]        
#                src_sent_str = ' '.join(src_sent_list) 
#                length = len(tokenizer.tokenize(src_sent_str))+2
#                token_struct_vec[i] = token_struct_vec[i][:length]
                            
    return skip, skip_reason, section_names, overall_sent_pos, sent_struct_vec, token_struct_vec


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
    abstract = ' '.join(abstract_sent_list).split()
    #abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
#    print("######abstract ",len(abstract),abstract)
    
    
    #sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    sents = doc_sent_list
    
    evaluated_1grams = [_get_word_ngrams(1, [sent.split()]) for sent in sents]
    #evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    #print("######predicted_1grams, len = ",len(evaluated_1grams),evaluated_1grams)
    reference_1grams = _get_word_ngrams(1, [abstract])
    #print("######gold_1grams, len =  ",len(reference_1grams),reference_1grams)
    evaluated_2grams = [_get_word_ngrams(2, [sent.split()]) for sent in sents]
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

        if args.base_LM.startswith('roberta'):
            #logger.info('Using roberta tokenizer')
            self.tokenizer = RobertaTokenizer.from_pretrained(args.base_LM)
        elif args.base_LM.startswith('bigbird-pegasus'):
            
            self.tokenizer = PegasusTokenizer.from_pretrained("google/"+args.base_LM, additional_special_tokens=['<unk_2>','<unk_3>','<unk_4>'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            #self.tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=True)

        if args.base_LM.startswith('roberta'):
            self.sep_token = '</s>'#2
            self.cls_token = '<s>'#0
            self.pad_token = '<pad>'#1  
            self.tgt_bos = ' madeupword0000 '
            self.tgt_eos = ' madeupword0001 '
            self.tgt_sent_split = ' madeupword0002 '
            self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.sep_token))[0]
            self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.cls_token))[0]
            self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.pad_token))[0]
            
        elif args.base_LM.startswith('bigbird-pegasus'):
            self.sep_token = '</s>'#1
            self.cls_token = '<s>'#2
            self.pad_token = '<pad>'#0  
            self.tgt_bos = '<unk_2>'
            self.tgt_eos = '<unk_3>'
            self.tgt_sent_split = '<unk_4>'
            self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.sep_token))[0]
            self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.cls_token))[0]
            self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.pad_token))[0]
            
            
            
        else:  
            self.sep_token = '[SEP]'
            self.cls_token = '[CLS]'
            self.pad_token = '[PAD]'
            self.tgt_bos = ' [unused0] '
            self.tgt_eos = ' [unused1] '
            self.tgt_sent_split = ' [unused2] '
            self.sep_vid = self.tokenizer.vocab[self.sep_token]
            self.cls_vid = self.tokenizer.vocab[self.cls_token]
            self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, doc, src, tgt, sent_labels,  is_test=False):
        
        init_logger(self.args.log_file)
        
        
        #skip empty document
        skip_reason=''
#        if ((not is_test) and len(src) == 0):
        
        if (len([sent for sent in src if sent!=''])==0):
            logger.info('Empty document is skipped.')
            logger.info(src)
            skip_reason='empty document'
            
            return None, skip_reason
        
        #get hierchical structural information 
        skip, skip_reason, section_names, _overall_sent_pos, _sent_struct_vec, _token_struct_vec = obtain_histruct_info(doc, self.args, self.tokenizer)
        
#        if ((not is_test) and skip):
        if skip:
            return None, skip_reason
        
        #list of sentences
        original_src_txt = src
            
        #sent_labels: a list of indices where the sentences should be included in the summary, label=1
        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1
        
        #get a list of indices of enough long sentences, use it to remove short sentences later
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        
        #remove short sentences 
        src = [src[i] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        sent_struct_vec = [_sent_struct_vec[i] for i in idxs]
        overall_sent_pos = [_overall_sent_pos[i] for i in idxs]
        if _token_struct_vec is None:
            token_struct_vec = None
        else:
            token_struct_vec = [_token_struct_vec[i] for i in idxs]
            
        print(len(_sent_labels))
        print(len(sent_labels))
        print(len(_sent_struct_vec))
        print(len(sent_struct_vec))
        
        #shorten long documents (remove last sentences), default: do not short, args.max_src_nsents=0
        if (self.args.max_src_nsents!=0):
            src = src[:self.args.max_src_nsents]
            sent_labels = sent_labels[:self.args.max_src_nsents]
            sent_struct_vec = sent_struct_vec[:self.args.max_src_nsents]
            overall_sent_pos = overall_sent_pos[:self.args.max_src_nsents]
            if token_struct_vec is not None:
                token_struct_vec = token_struct_vec[:self.args.max_src_nsents]
        
        #flat list
        if token_struct_vec is not None:
            token_struct_vec = sum(token_struct_vec,[]) 
       
        #skip too short documents if it is not test data
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            logger.info('Too short document (less than %d sentences) is skipped.'%(self.args.min_src_nsents))
            logger.info('length of original text %d'%len(original_src_txt))
            logger.info('length of text after removing short sentences %d'%len(src))
            skip_reason='too short document (less than %d sentences)'%self.args.min_src_nsents
            return None, skip_reason
        
      
        #preprocessed article_text, a list of sentences in the article
        src_txt = src#!!!
        #join sentences into text, add cls_token and sep_token between sentences
        if self.args.base_LM.startswith('roberta'):
            text = '{} {}'.format(self.sep_token, self.cls_token).join(src_txt)
        else:
            text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
            
        
        
        #tokenize using the tokenizer
        src_subtokens = self.tokenizer.tokenize(text)
        #add cls_token and sep_token at the beginning and the end of the text
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        
        
        #convert tokens to ids
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        
#        print("#################src_txt",len(src_txt),src_txt)
#        print("#################text",len(text), text)
#        print("#################src_subtokens",len(src_subtokens), src_subtokens)
#        print("#################src_subtokens_idxs",len( src_subtoken_idxs),  src_subtoken_idxs)
#        print("#################sent_labels",len(sent_labels), sent_labels)
       
       
        #segments_ids
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]    
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1] 
                
        #cls_ids, indices of cls_tokens  
        #print('src_subtoken_idxs',src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid] 
        #print('cls_ids',cls_ids)
        
       
        #sent_labels
        sent_labels = sent_labels[:len(cls_ids)]
        
#        print('#################segments_ids',len(segments_ids),segments_ids)
#        print("#################cls_ids",len(cls_ids), cls_ids)
#        print("#################sent_labels",len(sent_labels), sent_labels)
           
        
        #preprocessing of gold summaries
        
        #tgt_subtokens_str =  self.tgt_bos + self.tgt_sent_split.join([' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + self.tgt_eos
        tgt_subtokens_str =  self.tgt_bos + self.tgt_sent_split.join([' '.join(self.tokenizer.tokenize(tt)) for tt in tgt]) + self.tgt_eos
        
        
        #shorten long summaries      
        if self.args.max_tgt_ntokens!=0:
            tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        else:
            tgt_subtoken = tgt_subtokens_str.split()
            
        #skip if the summary is too short
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            logger.info('Skipped since the gold summary is too short')
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken) 
#        print("#################tgt_subtokens_str",len(tgt_subtokens_str), tgt_subtokens_str)
#        print("#################tgt_subtoken",len(tgt_subtoken), tgt_subtoken)
#        print("#################tgt_subtoken_idxs",len(tgt_subtoken_idxs), tgt_subtoken_idxs)
        
        #src_txt, tgt_txt
        tgt_txt = '<q>'.join(tgt) #plain text of gold summary, sentences joined by <q>
        src_txt = [original_src_txt[i] for i in idxs] #a list of sentences in the article
       
        #check
        if not len(sent_labels)==len(cls_ids)==len(sent_struct_vec):
            print(len(sent_labels),len(cls_ids),len(sent_struct_vec))
            print(sent_labels[:10],cls_ids[:10],sent_struct_vec[:10])
            for sent in src:
                if 0 in self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)):
                    print('!!!',sent)
                    
                    
            #print('src_subtoken_idxs',src_subtoken_idxs)
            #print('cls_ids',cls_ids)
            
        assert len(sent_labels)==len(cls_ids)==len(sent_struct_vec) #nr. of sentences
        if token_struct_vec is not None:
            assert len(segments_ids)==len(src_subtoken_idxs)==len(token_struct_vec) #nr. of tokens
        else:
            assert len(segments_ids)==len(src_subtoken_idxs)
            
        check_data(self.args, src_subtoken_idxs,token_struct_vec, segments_ids)
        
     
            

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, sent_struct_vec, token_struct_vec,overall_sent_pos,section_names

def check_data(args, src_subtoken_idxs,token_struct_vec, segments_ids):
    
    #subtokens to sentences 
    li=[]
    lists=[]
    
    SEP_IDX=0
    if (args.base_LM.startswith('bert')):
        SEP_IDX=102
    elif (args.base_LM.startswith('bigbird-pegasus')):
        SEP_IDX=1
    else:
        SEP_IDX=2
 
    for idx in src_subtoken_idxs:
    
        if not idx==SEP_IDX:#id of [SEP] or </s>
            li.append(idx)
        else:
            li.append(idx)
            lists.append(li)
            li=[]
            continue

    #token_struct_vec to sentences
    li2=[]
    lists2=[]
    count=0
    t=token_struct_vec
    if token_struct_vec is not None:
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
                
    #they contain same number of sentences 
    if token_struct_vec is not None:
        assert len(lists)==len(lists2)==len(lists3)
    else:
        assert len(lists)==len(lists3)
        
        
    #sentences at same index contain same number of items (sutokens and its structure vector)          
    l=[len(x) for x in lists]
    if token_struct_vec is not None:
        l2=[len(x) for x in lists2]
    l3=[len(x) for x in lists3]
    
    
    for i in range(len(l)):
        if token_struct_vec is not None:
            assert l[i]==l2[i]==l3[i]
        else:
            assert l[i]==l3[i]
    
    

def format_to_histruct(args):
    init_logger(args.log_file)
    datasets = ['train', 'valid', 'test']
    #create save folder if not exisits
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        logger.info('Save folder created.')
    else:
        if len(os.listdir(args.save_path))!= 0:
            text = input('Save folder already exisits and is not empty. Do you want to remove it and redo preprocessing (yes or no) ?')
            if text.lower()=='yes':
                shutil.rmtree(args.save_path)
                os.mkdir(args.save_path)
                logger.info('YES: Save folder removed and recreated.')
            else:
                logger.info('NO: Program stopped.')
                exit()
    
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):       
            real_name = json_f.split('/')[-1].split('\\')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        
        pool = Pool(args.n_cpus)
        
        for d in pool.imap(_format_to_histruct, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_histruct(params):
   
    
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    init_logger(args.log_file)
    logger.info('Using tokenizer: '+args.base_LM)
    if not args.obtain_tok_se:
            logger.info('Do not obtain token structure vectors')
    
    #check if the save file already exists
    if (os.path.exists(save_file)):
        text = input("Save file %s already exisits. Do you want to remove it and redo preprocessing (yes or no) ?"%(save_file))
        if text.lower()=='yes':
            os.remove(save_file)
            logger.info('YES: Save file removed.')
        else:
            logger.info('NO: Program stopped.')
            exit()
    
    
    bert = BertData(args)
    logger.info('#'*50)
    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file, encoding='utf-8')) #nr. of documents in one json file
    
    datasets = []
    if (args.summ_size!=0):
        logger.info("Do greedy selection to create oracle summaries, summary size:"+str(args.summ_size))
    else:
        logger.info("Do greedy selection to create oracle summaries, summary size: long")
    
    
    skip_reasons=[]  
    
    for d in jobs:
        #get list of source sentences and gold summary sentences (lowercase) and clean
        source, tgt = d['article_text'], d['abstract_text']
        
        source = [sent.strip().lower() for sent in source]
        
        tgt = [s.replace('<S>','').replace('</S>','').strip().lower() for s in tgt]
       
        #get index of selected sentences (in oracle summary)
        if (args.summ_size!=0):
            if (args.max_src_nsents!=0):
                sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, args.summ_size) 
            else:
                sent_labels = greedy_selection(source, tgt, args.summ_size) 
        else: 
            if (args.max_src_nsents!=0):
                sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, len(source[:args.max_src_nsents]))
            else:
                sent_labels = greedy_selection(source, tgt, len(source))
         
        tgt_sent_idx = sent_labels
                  
        b_data = bert.preprocess(d, source, tgt, sent_labels, is_test=is_test)

        if (b_data[0] is None):
            skip_reasons.append(b_data[1])         
            continue
        
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, sent_struct_vec, token_struct_vec,overall_sent_pos,section_names = b_data
                 
        b_data_dict = {"src": src_subtoken_idxs, ##!!
                       "tgt": tgt_subtoken_idxs, #!!
                       "src_sent_labels": sent_labels, ##!!
                       "segs": segments_ids, ##!!
                       'clss': cls_ids, ##!!
                       'src_txt': src_txt, ##!!
                       "tgt_txt": tgt_txt, ##!!
                       "tgt_sent_idx":tgt_sent_idx,#--
                       "overall_sent_pos":overall_sent_pos,#--
                       "sent_struct_vec":sent_struct_vec, ##!!
                       "token_struct_vec":token_struct_vec,
                       "section_names":section_names}##!!
        
        datasets.append(b_data_dict)
        if(len(datasets)%100==0):
            logger.info('----------------------------------Processed %d/%d'%(len(datasets),len(jobs)))

              
    #save and print skip reasons for later check           
    skip_reasons_dic ={}   
    if skip_reasons!=[]:   
        for r in set(skip_reasons):
            count=0
            for n in skip_reasons:
                if n==r:
                    count+=1
            skip_reasons_dic.update({r:(count,round(count/len(jobs),4))})    
            
    file_path = args.save_path+'/skip_reasons'
    file_name = file_path+'/'+ '.'.join(json_file.split('/')[-1].split('.')[:-1])+'.skip_reasons.txt'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    dic ={'file_name':json_file, 'nr. of total doc':len(jobs),'nr. of processed doc':len(datasets),
          'nr. of skipped instances':len(skip_reasons), 'skip percentage':round(len(skip_reasons)/len(jobs)*100,2),
          'skip_reasons':skip_reasons_dic}      
    with open(file_name, 'w+') as save:
            save.write(json.dumps(dic))        
    logger.info('File %s'%json_file)        
    logger.info('There are %d instances.'%len(jobs))
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Skipped instances %d, %f percentage of the %d instances' % (len(skip_reasons), round(len(skip_reasons)/len(jobs)*100,2), len(jobs)))
    logger.info('Skip reasons: %s'%(skip_reasons_dic))
    
    
    #save preprocessed dataset
    logger.info('Saving to %s' % save_file)   
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    
    
    


def merge_data_splits(args):
    
    save_path = '/'.join(args.save_path.split('/')[:-1])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
     
    corpora = {'train': None, 'valid': None, 'test': None}  
    #read datasets
    logger.info('Reading train/valid/test datasets...')       
    for f in glob.glob(pjoin(args.raw_path, '*.txt')):   
        
        data_type = f.replace(args.raw_path,'')[1:].replace('.txt','')
        if data_type=='val':
            data_type='valid'
        data = [json.loads(line) for line in open(f,'r',encoding='utf-8')]
        corpora[data_type]=data
    
    logger.info('There are %s / %s / %s documents in train/valid/test datasets.'% (len(corpora['train']),len(corpora['valid']),len(corpora['test']))) 
    
    #save_statistics
    stat={'nr. docs':(len(corpora['train']),len(corpora['valid']),len(corpora['test']))}
    stat_path = args.save_path.split('/')[0]+'/statistics.json'
    with open(stat_path, 'w+') as save:
                save.write(json.dumps(stat))
 
    logger.info('Merging documents...')  
    
    #merge data splits
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in a_lst: #pool.imap_unordered(_merge_data_splits, a_lst):#
            #print(d[0])
            #doc=json.load(open(d[0], encoding='utf-8'))
            dataset.append(d[0])
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

def obtain_section_names(args):
    
    save_file1 = args.save_path+'/unique_section_names.json'
    save_file2 = args.save_path+'/section_names_count.json'
   
    init_logger(args.log_file)
    logger.info("Obtaining section names...")
    
    section_names = []
   
    for f in glob.glob(pjoin(args.raw_path, '*.txt')):
        data = [json.loads(line) for line in open(f,'r',encoding='utf-8')]
        for doc in data:
            doc_sec_names = doc['section_names']
            doc_sec_names = [re.sub(r'[^a-zA-Z ]', '', s).lower().strip() for s in doc_sec_names]
            section_names.append(doc_sec_names)
            
    flat_sec_names = sum(section_names, [])
    unique_sec_names =sorted(set(flat_sec_names))
    nr = len(unique_sec_names)
    dic ={}
    for name in unique_sec_names:
        count=0
        for n in flat_sec_names:
            if n==name:
                count+=1
        dic.update({name:count})
    
    sorted_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
   
    logger.info("There are %i unique section names in the dataset."%(nr))
    logger.info("20 most frequent section names are:")
    logger.info(str(sorted_dic[:20]))

    with open(save_file1, 'w+') as save:
        save.write(json.dumps(unique_sec_names))
    with open(save_file2, 'w+') as save:
        save.write(json.dumps(sorted_dic))
        
    logger.info("DONE")
    

from transformers import LongformerModel, LongformerTokenizer
###############################################################################
def encode_section_names(args):
    init_logger(args.log_file)
    with open(args.raw_path+'/unique_section_names.json', encoding='utf-8') as file:
        section_names = json.load(file)
    logger.info('Encoding section names...')
    logger.info('There are %d unique section names in the dataset %s'%(len(section_names),args.dataset))
    logger.info('Section names embeddings combination mode: %s'%(args.sn_embed_comb_mode))
    
    if args.base_LM.startswith('longformer'):
        model = LongformerModel.from_pretrained('allenai/'+args.base_LM, cache_dir=args.temp_dir)  
        model.eval()
        tokenizer = LongformerTokenizer.from_pretrained('allenai/'+args.base_LM)
        
        section_names_embed={}
        
        for section_name in section_names:           
            input_ids = torch.tensor(tokenizer.encode(section_name)).unsqueeze(0)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
            global_attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)#do global attention everywhere
            outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
            if args.sn_embed_comb_mode=='sum':
                embed = torch.sum(outputs.last_hidden_state,dim=1).squeeze().tolist()
            elif args.sn_embed_comb_mode=='mean':
                embed = torch.mean(outputs.last_hidden_state,dim=1).squeeze().tolist()
            
            section_names_embed.update({section_name:embed})
            logger.info('section name encoded: %s, (%d/%d) '%(section_name, len(section_names_embed),len(section_names)))
        
        base_lm_name = args.base_LM.split('-')[0]+args.base_LM.split('-')[1][0].upper()
        path = args.save_path+'/section_names_embed_'+base_lm_name+'_'+args.sn_embed_comb_mode+'.pt'
        torch.save(section_names_embed,path)
        logger.info('DONE! Section names embeddings are saved in '+path)
###############################################################################
        
    

def compute_statistics(args):
    
    stat_path = args.raw_path+'/statistics.json'
   
    init_logger(args.log_file)
    logger.info("Computing statistics...")
    
    doc_len_para=[]
    doc_len_sent=[]
    doc_len_word=[]  
    summ_len_sent=[]
    summ_len_word=[]
    novel_2grams=[]
    novel_1grams=[]
    
    for f in glob.glob(pjoin(args.raw_path, '*.txt')):
        data = [json.loads(line) for line in open(f,'r',encoding='utf-8')]
        for doc in data:
        
            #doc=json.load(open(f, encoding='utf-8'))
            
            doc_len_para.append(len(doc["sections"]))
            doc_len_sent.append(len(doc["article_text"]))
            summ_len_sent.append(len(doc["abstract_text"]))
            
            src = doc["article_text"]
            tgt = [s.replace('<S>','').replace('</S>','') for s in doc['abstract_text']]
            flat_src = ''.join(src) 
            flat_summ = ''.join(tgt) 
            doc_len_word.append(len(flat_src.split())) 
            summ_len_word.append(len(flat_summ.split()))

            re = get_novel_ngrams_percentage(flat_src, flat_summ, 2) 
            if re is not None:
                novel_2grams.append(re)
            re = get_novel_ngrams_percentage(flat_src, flat_summ, 1) 
            if re is not None:
                novel_1grams.append(re)
            
            

        
    stat = {'avg. doc length(words)': round(statistics.mean(doc_len_word),2), 
            'min. doc length(words)': min(doc_len_word), 
            'max. doc length(words)': max(doc_len_word),    
           'avg. doc length(sentences)': round(statistics.mean(doc_len_sent),2), 
           'min. doc length(sentences)': min(doc_len_sent), 
           'max. doc length(sentences)': max(doc_len_sent), 
           'avg. doc length(paragraphs)':round(statistics.mean(doc_len_para),2),
           'min. doc length(paragraphs)':min(doc_len_para),
           'max. doc length(paragraphs)':max(doc_len_para),
           'avg. summary length(words)': round(statistics.mean(summ_len_word),2), 
           'min. summary length(words)': min(summ_len_word), 
           'max. summary length(words)': max(summ_len_word), 
           'avg. summary length(sentences)': round(statistics.mean(summ_len_sent),2), 
           'min. summary length(sentences)': min(summ_len_sent),
           'max. summary length(sentences)': max(summ_len_sent),
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
    
    if len(summ_ngrams)==0 or len(src_ngrams)==0:
        return None
    else:
        same = len(set(summ_ngrams).intersection(set(src_ngrams)))
        novel = len(summ_ngrams)-same
        return round((novel/len(summ_ngrams))*100,2)


        

#def extract_histruct_items(args):
#    #tok_sent_dir = os.path.abspath(args.tok_sent_path)
#    tok_para_dir = os.path.abspath(args.tok_para_path)
#    histruct_dir = os.path.abspath(args.histruct_path)
#    
#    if not os.path.exists(histruct_dir):
#        os.makedirs(histruct_dir)
#        
#    init_logger(args.log_file)
#    logger.info("Extracting histruct items...")
#    
#    for f in glob.glob(pjoin(args.tok_sent_path, '*.json')):
#       
#        real_name = f.split('/')[-1].split('\\')[-1]
#        histruct_story_path = histruct_dir+'\\'+real_name
#        
#        
#        source, tgt = obtain_source_summ(f, args.lower)
#        src_para_tokens = obtain_para(f,args.lower,tok_para_dir,real_name)
#        doc = {'src': source, 'tgt':tgt, 'src_para_tokens':src_para_tokens}
#
#        with open(histruct_story_path, 'w+') as save:
#                save.write(json.dumps(doc))
#    logger.info("DONE")
        
    
    

#obtain list of sentences and list of paragraphs from source text
#the results are kept to obtain hierarchical positions
#def obtain_para(f,lower,tok_para_dir,real_name):
#    
#    para_story_path = tok_para_dir+'\\'+real_name #same name
#    #print("###############sent_story_path",sent_story_path)
#    #print("###############para_story_path",para_story_path)
#    
#    #get list of tokens of sentences from source text
##    source_sent_tokens=[]
##    flag =False
##    for sent in json.load(open(sent_story_path, encoding='utf-8'))['sentences']:
##        tokens = [t['word'] for t in sent['tokens']]
##        if (lower):
##            tokens = [t.lower() for t in tokens]
##        if(tokens[0] == '@highlight'):
##            flag = True
##            #print("tokens",tokens)
##            #print("flag",flag)
##            continue
##        if (not flag):
##            source_sent_tokens.append(tokens)
#    #print (len(source_sent_tokens),source_sent_tokens)
#    
#    #get list of tokens of paragraphs from source text
#    source_para_tokens=[]
#    tgt_para_tokens=[]
#    flag =False
#    for sent in json.load(open(para_story_path, encoding='utf-8'))['sentences']:
#        tokens = [t['word'] for t in sent['tokens']]
#        if (lower):
#            tokens = [t.lower() for t in tokens]
#        if(tokens[0] == '@highlight'):
#            flag = True
#            tgt_para_tokens.append([])
#            continue
#        if (flag):
#            tgt_para_tokens[-1].extend(tokens)
#        else:
#            source_para_tokens.append(tokens)
#            
#    source_para_tokens = [clean(' '.join(para)).split() for para in source_para_tokens]        
#    
#    return source_para_tokens
#
#def obtain_source_summ(p, lower):
#    source = []
#    tgt = []
#    flag = False
#    for sent in json.load(open(p, encoding='utf-8'))['sentences']:
#        tokens = [t['word'] for t in sent['tokens']]
#        if (lower):
#            tokens = [t.lower() for t in tokens]
#        if (tokens[0] == '@highlight'):
#            flag = True
#            tgt.append([])
#            continue
#        if (flag):
#            tgt[-1].extend(tokens)
#        else:
#            source.append(tokens)
#
#    source = [clean(' '.join(sent)).split() for sent in source]
#    tgt = [clean(' '.join(sent)).split() for sent in tgt]
#    return source, tgt
#
#
#def tokenize(args):
#    stories_dir = os.path.abspath(args.raw_path)
#    tok_sent_dir = os.path.abspath(args.tok_sent_path)
#    tok_para_dir = os.path.abspath(args.tok_para_path)
#    init_logger(args.log_file)
#    
#    
#    if os.path.exists(tok_sent_dir):
#        logger.info("SENT - Save path %s already exisits, remove it!" % (tok_sent_dir))
#        shutil.rmtree(tok_sent_dir)
#    if os.path.exists(tok_para_dir):
#        logger.info("PARA - Save path %s already exisits, remove it!" % (tok_para_dir))
#        shutil.rmtree(tok_para_dir)
#        
#        
#        
#    logger.info("Preparing to tokenize %s" % (stories_dir))    
#    stories = os.listdir(stories_dir)
#    # make IO list file
#    logger.info("Making list of files to tokenize...")
#    with open("mapping_for_corenlp.txt", "w") as f:
#        for s in stories:
#            if (not s.endswith('story')):
#                continue
#            f.write("%s\n" % (os.path.join(stories_dir, s)))
#
##    #split sentences
##    command_sent = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
##               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
##               'json', '-outputDirectory', tok_sent_dir]
##    
##    #split paragraphs
##    command_para = [corenlp_path, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
##               '-ssplit.eolonly', 'true', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
##               'json', '-outputDirectory', tok_para_dir]#
#    
#    #split sentences
#    command_sent = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
#               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
#               'json', '-outputDirectory', tok_sent_dir]
#    
#    #split paragraphs
#    command_para = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
#               '-ssplit.eolonly', 'true', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
#               'json', '-outputDirectory', tok_para_dir]#
#    
#    logger.info("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_sent_dir))
#    subprocess.call(command_sent,shell=True)
#    logger.info("SENT - Stanford CoreNLP Tokenizer has finished.")
#    #os.remove("mapping_for_corenlp.txt")
#    
#    logger.info("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tok_para_dir))
#    subprocess.call(command_para,shell=True)#
#    logger.info("PARA - Stanford CoreNLP Tokenizer has finished.")
#    os.remove("mapping_for_corenlp.txt")
#
#    # Check that the tokenized stories directory contains the same number of files as the original directory
#    num_orig = len(os.listdir(stories_dir))
#    num_tokenized_para = len(os.listdir(tok_para_dir))
#    num_tokenized_sent = len(os.listdir(tok_sent_dir))
#    if num_orig != num_tokenized_sent:
#        raise Exception(
#            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
#                tok_sent_dir, num_tokenized_sent, stories_dir, num_orig))
#    if num_orig != num_tokenized_para:
#        raise Exception(
#            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
#                tok_para_dir, num_tokenized_para, stories_dir, num_orig))
#    logger.info("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_para_dir))
#    logger.info("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tok_sent_dir))