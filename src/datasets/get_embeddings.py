import pandas as pd
import ast
import csv
import json
import time
from prompts import template_map, class_map

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

dataset_list = [
 'gtsrb',
 'cifar-10',
 'caltech-101',
 'cifar-100',
 'country211',
 'fgvc-aircraft-2013b-variants102',
 'oxford-flower-102',
 'food-101',
 'kitti-distance',
 'mnist',
 'patch-camelyon',
 'resisc45_clip',
 'stanford-cars',
 'voc-2007-classification',
 'oxford-iiit-pets',
 'eurosat_clip',
 'hateful-memes',
 'rendered-sst2',
 'dtd',
 'fer-2013',
]

def get_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        avg_embeddings = torch.mean(embeddings, 0)
        
    return avg_embeddings


if __name__ == "__main__":

    # Import our models. The package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = model.to("cuda")

    knowledge_dict = []

    for dataset in dataset_list:
        external_path = f"external/{dataset}_knowledge.tsv"
        gpt3_path = f"gpt3/GPT3_{dataset}.tsv"
        external_list = json.load(open(external_path, encoding='utf-8'))
        gpt3_list = json.load(open(gpt3_path, encoding='utf-8'))

        for external_item, gpt3_item in zip(external_list, gpt3_list):
            item = {}
            
            item['dataset'] = dataset
            item['classname'] = external_item['classname']
            item['wiki'] = external_item['def_wiki']
            item['wn'] = external_item['def_wn']
            item['gpt3'] = gpt3_item['gpt3']
            item['template'] = template_map[dataset]
            knowledge_dict.append(item)

    embedding_list = []

    count = 0
    start_time = time.time()
    for item in knowledge_dict:
        emb_item = {}
        count += 1
        templates = item['template']
        classname = item['classname']
        wiki = item['wiki']
        gpt3 = item['gpt3']
        
        # get text embedding
        texts = []
        for template in templates:
            texts.append(template.replace('{}', classname))
        texts_embeddings = get_embeddings(texts)

        # get wiki embedding
        if wiki is None or len(wiki) == 0:
            wiki_embeddings = texts_embeddings
        else:
            wiki_des_texts = []
            for text in texts:
                if type(wiki) is str:
                    wiki_des_texts.append(text + " " + wiki)
                else:
                    for wiki_ in wiki:
                        wiki_des_texts.append(text + " " + wiki_)
            wiki_embeddings = get_embeddings(wiki_des_texts)
    #         for t in wiki_des_texts:
    #             print(t)
    #         print(len(wiki_des_texts))
    #         print('good')
            
        if gpt3 is None or len(gpt3) == 0:
            gpt3_embeddings = texts_embeddings
        else:
            gpt3_des_texts = []
            for text in texts:
                if type(gpt3) is str:
                    gpt3_des_texts.append(text + " " + gpt3)
                else:
                    for gpt3_ in gpt3:
                        gpt3_des_texts.append(text + " " + gpt3_)
            gpt3_embeddings = get_embeddings(gpt3_des_texts)    
    #         for t in gpt3_des_texts:
    #             print(t)
    #         print(gpt3_des_texts)
    #         print(good)
    #
        time_temp = time.time()
        emb_item['dataset'] = dataset
        emb_item['classname'] = external_item['classname']
        emb_item['text_emb'] = texts_embeddings
        emb_item['wiki_text_emb'] = wiki_embeddings
        emb_item['gpt3_text_emb'] = gpt3_embeddings
        embedding_list.append(emb_item)
        if count % 100 == 0:
            print(f"finish {count}, since start, require {time_temp - start_time}")

        torch.save(embedding_list, 'sentence_embeddings.pt')