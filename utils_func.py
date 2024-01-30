import torch
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import argparse
import random
import pickle
from tqdm import tqdm
import numpy as np
from ast import literal_eval
import re
from wiki_api.wikidata import id2entity

from wiki_api.strings import question_token

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning MH Model Editing")
    parser.add_argument('--device', type=str, default = "cuda")
    parser.add_argument('--model',type=str, default='EleutherAI/gpt-j-6B')
    parser.add_argument('--dataset',type=str, default='MQuAKE-CF-3k')
    parser.add_argument('--relation_path',type=str, default='data/relation.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_return', type=int, default=2)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=10)
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--starting_line', type=int, default=0)
    parser.add_argument('--beam_width', type=int, default=2)
    parser.add_argument('--loss', type=str, default='prob_div_log')
    parser.add_argument('--mode', type=str, default='beam')
    parser.add_argument('--template', type=bool, default=True)
    parser.add_argument('--NatureL', type=bool, default=True)
    parser.add_argument('--correctConflict', type=bool, default=True)
    parser.add_argument('--template_number', type=int, default=3)
    parser.add_argument('--entropy_template_number', type=int, default=6)
    args = parser.parse_args()

    return args

args = parse_args()

ner_tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
ner_model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")


def match_func(ground_truth_fact, fact_str, mode):
    
    
        if fact_str == '':
            return 0,0

        part_match_cor, exact_match_cor = 0,0
        for fact in ground_truth_fact:
            if fact in fact_str:
                part_match_cor =1
                print(f"{mode}-->question: Part fact match")
                break
        
        ext = 0    
        for fact in ground_truth_fact:
            if fact in fact_str:
                ext+=1 
        if ext == len(ground_truth_fact) :
            exact_match_cor =1
            print(f"{mode}-->question: Exact fact match")
            
        # test whether the pruning is working
        if mode == 'Pruned' and exact_match_cor == 1:
            redunant_detected = 0
            fact_str = fact_str[:-1] #remove the last period
            input_fact_list = fact_str.split('.\n')
            #print("input_fact_list", input_fact_list)
            for fact in input_fact_list:
                if fact not in ground_truth_fact:
                    redunant_detected = 1
            if redunant_detected >0:
                print("Failed Pruning")
                      
        # if mode == 'raw':
        #     if ext == len(ground_truth_fact) :
        #         exact_match_cor =1
        #         print(f"{mode}-->{i}-case-{j}-question: Exact fact match")
        # elif mode =='prun':
        #     fact_list = fact_str.split('.\n')
        #     print("fact_list", fact_list)
        #     if ext == len(ground_truth_fact) and ext == len(fact_list) :
        #         exact_match_cor =1
        #         print(f"{mode}-->{i}-case-{j}-question: Exact fact match")
        #     #break
            
        return part_match_cor, exact_match_cor
    

def QA_func(model, tokenizer, line, input_fact, question, ans_prompt, mode):
    
    correct_ans = 0
    if input_fact == '':
        input_fact_str = input_fact
    else:
        input_fact = input_fact[:-1] #remove the last period
        input_fact_list = input_fact.split('.\n') # turn the str into a list
        input_fact_str = ', '.join(input_fact_list) # add the comma in the sentences 
    prom_text = ans_prompt + "Given fact: " + input_fact_str + ', '+ question + '\nAnswer:'
    #print("prom_text", prom_text)
    input_ids = tokenizer.encode(prom_text, return_tensors="pt").to(args.device)
    output = model.generate(input_ids, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, temperature=args.temp, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).replace("\n", ", ")
    simple_ground_ans = re.sub(r"[^a-zA-Z ]+", '', line['new_answer']).lower()
    simple_ans = re.sub(r"[^a-zA-Z ]+", '', ans).lower()
    if simple_ground_ans in simple_ans:
        correct_ans = 1
    else: 
        for alias in line['new_answer_alias']:
            simple_alias = re.sub(r"[^a-zA-Z ]+", '', alias).lower()
            if simple_alias in simple_ans:
                correct_ans = 1
            break
    print(f"{mode}:{correct_ans}-->Edited Ans:'{line['new_answer']}/{simple_ground_ans}', Our Ans:'{ans}/{simple_ans}'")
    
    return simple_ans, correct_ans
    
def load_orig_entity(orig_triplets_dict, eid, pid, entity):

    original_eid_facts = find_lines_by_entity(orig_triplets_dict, eid)
    for orig_fact in original_eid_facts:
        orig_eid, orig_pid, tail_entity = orig_fact.split('\t')
        if pid == orig_pid:
                return tail_entity
            
    return entity

def ner_entity(question):
    
    ner_results = nlp(question)
    
    if len(ner_results) == 1:
        target_entity = ner_results[0]['word']
        return target_entity
    elif len(ner_results) >= 1:
        first_target_entity = ner_results[0]['word']
        the_first_target_entity = 'the' + first_target_entity
        new_question = question.replace(first_target_entity, the_first_target_entity)
        target_entity = nlp(question)[0]['word']
        return target_entity
    else:
        return 'xxxx'



def load_dataset(path):
    
    with open(path, "r") as f:
        lines = json.load(f)
    
    return lines

def load_prompt(path):
    
    with open(path, "r") as f:
        text = f.read()
        
    return text

def load_KG(path):
    
    content_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            content_list.append(line.strip())
            
    return content_list

def load_train_question(path):
    
    content_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            content_list.append(literal_eval(line.strip()))
            
    return content_list

def load_triplets_dict(filename):
    with open(filename, 'rb') as file:
        triplets_dict = pickle.load(file)
    return triplets_dict


def fact_triplet_to_sentence(triplet, relation_dict):
    
    subject_id, predicate_id, object_id = triplet.split('\t')

    subject = id2entity(subject_id)
    predicate = relation_dict[predicate_id]
    object_ = id2entity(object_id)

    sentence = f"{subject} {predicate} {object_}"
    
    return sentence

def build_relation(lines, NL_dict):
    
    relation_dict = dict()
    if NL_dict:
        for i in lines["results"]['bindings']:
            if f"{i['property']['value'][31:]}" in NL_dict.keys():
                relation_dict[f"{i['property']['value'][31:]}"] = NL_dict[f"{i['property']['value'][31:]}"]
            else:
                relation_dict[f"{i['property']['value'][31:]}"] = i['propertyLabel']['value']
    else:
        for i in lines["results"]['bindings']:
            relation_dict[f"{i['property']['value'][31:]}"] = i['propertyLabel']['value']
        
    revserse_dict = {v:k for k,v in relation_dict.items()}
    
    return relation_dict, revserse_dict
    

def find_lines_by_entity(triplets_dict, entity_number):
    # Search for lines containing the given entity number
    if entity_number in triplets_dict:
        return triplets_dict[entity_number]
    return []

def build_fact(lines):

    edit_triplets_list = set()
    for d in lines:
        for r in d["orig"]["edit_triples"]:
            head_e, rel, tail_e = r
            edit_triplets_list.add((head_e,rel))

    return  edit_triplets_list


def build_rome_fact(lines):
    """_summary_

    Args:
        lines (list): 

    Returns:
        prompts,  ground_truth, target_new, subjects
    """
    
    prompts = []
    ground_truth = []
    target_new = []
    subjects = []
    
    for d in lines:
        for r in d["requested_rewrite"]:
            prompts.append(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
            ground_truth.append(f'{r["target_true"]["str"]}')
            target_new.append(f'{r["target_new"]["str"]}')
            subjects.append(f'{r["subject"]}')

    return prompts, ground_truth, target_new, subjects


def build_question(lines):
    
    questions = set()
    for d in lines:
        r =random.choice(d["questions"])
        questions.add(f'{r}')
        
    questions = list(questions)
    
    return questions



def retrieve_facts(query, fact_embs, contriever, tok, fact_number=1):
        inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to(contriever.device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
        sim = (query_emb @ fact_embs.T)[0]
        knn = sim.topk(fact_number, largest=True)
        return knn.indices

    
def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
        all_embs = []
        for i in tqdm(range(0, len(sents), BSZ)):
            sent_batch = sents[i:i+BSZ]
            inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(contriever.device)
            with torch.no_grad():
                outputs = contriever(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            all_embs.append(embeddings.cpu())
        all_embs = torch.vstack(all_embs)
        return all_embs
    
def retr_fact(input, embs, new_facts_set, fact_number, contriever, contriever_tokenizer):
    
    fact_ids = retrieve_facts(input, embs, contriever, contriever_tokenizer, fact_number)
    selected_fact = [new_facts_set[fact_id] for fact_id in fact_ids]
    
    return selected_fact 

def retr_relations(facts, relation_dict):
    
    '''
    retriveve all the relations within a fact list like ['Q596874\P155\Q3205815', 'Q596874\P123\Q921536', 'Q596874\P136\Q1543778']
    return relation_id and relation_names in list format
    '''
    
    relation_ids = set()
    relation_names = set()
    for fact in facts:
        id = fact.split('\t')[1]
        relation_ids.add(id)
        try:
            name = relation_dict[id]
            relation_names.add(name)
        except:
            pass
        
    
    return  list(relation_names), list(relation_ids)


def get_k_candidates(model, tokenizer, input_text, new_tokens = 50,  num_candidates=10, top_k=10, temperature=1):
    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(args.device)
    
    # Generate multiple answer candidates
    generated_candidates = model.generate(
        input_ids,
        max_new_tokens=new_tokens,  # Adjust the value based on your desired length of generated answers
        num_return_sequences=num_candidates,  # Adjust the number of answer candidates you want to generate
        do_sample=True,
        top_k=top_k,
        temperature=temperature
    )
    decoded_candidates = [tokenizer.decode(candidate, skip_special_tokens=True) for candidate in generated_candidates]

    return decoded_candidates
        
        
def construct_examples(input_question, k, train, question_embs, contriever, contriever_tokenizer):
    
    icl_examples = ""
    #train_ids = np.random.randint(len(train), size=k)
    train_ids = retrieve_facts(input_question, question_embs, contriever, contriever_tokenizer, k)
    for train_id in train_ids:
        line = train[train_id]
        new_fact = "Given fact:"
        for num, r in enumerate(line["new_single_hops"]):
             fact = f"{r['cloze']} {r['answer']}"
             if num ==0:
                new_fact = new_fact + fact
             else:
                new_fact = new_fact +', ' + fact
        questions = random.choice(line['questions'])
        target_new = line['new_answer']
        icl_examples += f'{new_fact}, {questions} {target_new}.'+'\n\n'
    
    return icl_examples


def construct_extraction_examples(input_question, k, train, question_embs, contriever, contriever_tokenizer, NatureL):
    
    NL_dict = load_dataset('prompts/templates/cloze_templates_NL.json')
    icl_examples = ""
    #train_ids = np.random.randint(len(train), size=k)
    train_ids = retrieve_facts(input_question, question_embs, contriever, contriever_tokenizer, k)
    for train_id in train_ids:
        line = train[train_id]
        new_fact = ""
        
        for num, r in enumerate(line["orig"]["new_triples_labeled"]):
            if NatureL:
                fact = " ".join([r[0], NL_dict[line["orig"]["new_triples"][num][1]], r[2]])+'.'
            else:
                fact = " ".join(r)+'.'
            if num ==0:
                new_fact = new_fact + fact
            else:
                new_fact = new_fact +' ' + fact
        questions = random.choice(line['questions'])
        #target_new = line['new_answer']
        icl_examples += f'Question: {questions} Answer: {new_fact}'+'\n'
        #icl_examples += f'Given question: {questions}. The answer is: {new_fact}.'+'\n\n'
    
    return icl_examples


def eval(model, tokenizer, icl_examples, targets, x):
    ppls = [] 
    for target in targets:
        tgt_len = len(tokenizer.encode(' ' + target))
        #print(''.join(icl_examples) + f'{x}? {target}')
        encodings = tokenizer(''.join(icl_examples) + f'{x}? {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(args.device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl.item())
    return ppls


def sequences_prob(model, tokenizer, input_text, args):
    
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    model.eval
    with torch.no_grad():
        logits = model(input_tokens).logits
    sentence_probability = 1
    num = input_tokens.shape[1]
    for i, ids in enumerate(*input_tokens):
        id_probability = torch.softmax(logits[:,i,:], dim=-1)[:, ids]
        #sentence_probability = sentence_probability*torch.pow(id_probability, -num)
        #print("id_probability**(-num)", id_probability, num, id_probability**(1./num))
        sentence_probability = sentence_probability*(id_probability**(1./num))
    
    return sentence_probability


    
    
    
    