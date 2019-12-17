import torch
from torch.nn import CrossEntropyLoss
from transformers import *
import numpy as np
import math
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import os
import sys
logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
def uni_predict(text, model, tokenizer):
    # Tokenized input
    # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
    text = text
    tokenized_text = tokenizer.tokenize(text)
    sentence_score = 0
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    length = len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    #masked_tensor = torch.tensor([masked_index])
    with torch.no_grad():
        outputs = model(tokens_tensor, labels= tokens_tensor)
    loss = outputs[0]
    sentence_score = -loss
    return sentence_score

def predict(text, bert_model, bert_tokenizer):
    # Tokenized input
    # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
    text = "[CLS] " + text + " [SEP]" #special token for BERT, RoBERTa
    tokenized_text = bert_tokenizer.tokenize(text)
    sentence_score = 0
    length = len(tokenized_text)-2
    for masked_index in range(1,len(tokenized_text)-1):
        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_word = tokenized_text[masked_index]
        #tokenized_text[masked_index] = '<mask>' #special token for XLNet
        tokenized_text[masked_index] = '[MASK]' #special token for BERT, RoBerta
        # Convert token to vocabulary indices
        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        index = torch.tensor(tokenizer.convert_tokens_to_ids(masked_word))
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to('cuda')
        index = index.to('cuda')
        #masked_tensor = torch.tensor([masked_index])
        with torch.no_grad():
            outputs = bert_model(tokens_tensor)
        prediction_scores = outputs[0]
        prediction_scores = prediction_scores.view(-1, model.config.vocab_size)
        prediction_scores = prediction_scores[masked_index].unsqueeze(0)
        loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
        masked_lm_loss = loss_fct(prediction_scores, index.view(-1))
        tokenized_text[masked_index] = masked_word
        sentence_score -= masked_lm_loss.item()
        tokenized_text[masked_index] = masked_word
    sentence_score = sentence_score/length
    return sentence_score

def xlnet_predict(text, model, tokenizer):
    tokenized_text = tokenizer.tokenize(text)
    # text = "[CLS] Stir the mixture until it is done [SEP]"
    sentence_score = 0
    #Sprint(len(tokenized_text))
    for masked_index in range(0,len(tokenized_text)):
        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_word = tokenized_text[masked_index]
        masked_word = tokenized_text[masked_index]
        tokenized_text[masked_index] = '<mask>'
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_text)).unsqueeze(0)
        index = torch.tensor(tokenizer.convert_tokens_to_ids(masked_word))
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        perm_mask[:, :, masked_index] = 1.0  # Previous tokens don't see last token
        target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
        target_mapping[0, 0, masked_index] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
        input_ids = input_ids.to('cuda')
        perm_mask = perm_mask.to('cuda')
        target_mapping = target_mapping.to('cuda')
        index = index.to('cuda')
        with torch.no_grad():
            outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels = index)
        next_token_logits = outputs[0]
        length = len(tokenized_text)
        # predict_list = predictions[0, masked_index]
        sentence_score -= next_token_logits.item()
        tokenized_text[masked_index] = masked_word
    return sentence_score/(length)

test = sys.argv[1]
model_type = sys.argv[2]
robust = sys.argv[3]
if model_type=='xlnet':
    # For XLNet
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
elif model_type=='bert':
    # For BERT
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForMaskedLM.from_pretrained('bert-large-uncased')
elif model_type=='roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForMaskedLM.from_pretrained('roberta-large')
elif model_type=='gpt':
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
elif model_type=='gpt-2':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

model.to('cuda')
model.eval()

if robust=='r':
    with open("CATS/Robust_commonsense_test/{}.txt".format(test), "r") as f:
        file = f.readlines()
    num = len(file)
    count = 0
    curr = 0
    for line in file:
        line = line.strip().split("\001")
        label_1 = int(line[0])
        label_2 = int(line[3])
        score_list = []
        for sentence in line:
            if not len(sentence)==1:
                if model_type=='xlnet':
                    score = xlnet_predict(sentence, model=model, tokenizer=tokenizer)
                elif model_type=='bert' or model_type=='roberta':
                    score = predict(sentence, bert_model=model, bert_tokenizer=tokenizer)
                else:
                    score = uni_predict(sentence, model=model, tokenizer=tokenizer)
                score_list.append(score)
        #print(score_list)
        score_list_1 = score_list[:2]
        score_list_2 = score_list[2:]
        predict_label_1 = score_list_1.index(max(score_list_1))
        predict_label_2 = score_list_2.index(max(score_list_2))
        if predict_label_1==label_1 and predict_label_2==label_2:
            count += 1
        elif predict_label_1!=label_1 and predict_label_2!=label_2:
            count += 1
        curr += 1
        #print (count, curr, count/curr)
    print(test+' '+model_type+':-------------------')
    print (count/num)
else:
    with open("CATS/commonsense_ability_test/{}.txt".format(test), "r") as f:
        file = f.readlines()
    num = len(file)
    count = 0
    curr = 0
    for line in file:
        line = line.strip().split("\001")
        label = int(line[0])
        score_list = []
        for sentence in line[1:]:
            if model_type=='xlnet':
                score = xlnet_predict(sentence, model=model, tokenizer=tokenizer)
            elif model_type=='bert' or model_type=='roberta':
                score = predict(sentence, bert_model=model, bert_tokenizer=tokenizer)
            else:
                score = uni_predict(sentence, model=model, tokenizer=tokenizer)
            score_list.append(score)
        print(score_list)
        predict_label = score_list.index(max(score_list))
        print(predict_label, label)
        if predict_label==label:
            count += 1
        curr += 1
        print (count, curr, count/curr)
    print(test+' '+model_type+':-------------------')
    print (count/num)