#Add the function correct_self_confliction
import torch
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelWithLMHead
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
from tqdm import tqdm
import numpy as np

from utils_func import  *
from wiki_api.strings import question_token
from wiki_api.Wiki import OnelineSearchEngine
wikisearch = OnelineSearchEngine()
from model import Extract, Prune 


def tempalate_extractor(question, NatureL, mode):
    
    icl_examples = ""
    
    question_set = question_token(question)
    score_list = [len(question_set & tuple[3]) / len(question_set | tuple[3]) for tuple in tuple_list]
    # index = score_list.index(max(score_list))
    # template_id = tuple_list[index][-1]
    order = list(np.argsort(score_list)[-20:-1]) # the last one is most relevant
    order.reverse() # reverse the list
    founded_tuple = [tuple_list[i] for i in order]
    #print("founded_tuple", founded_tuple)
    if mode == 'prob':
        template_number = 0
        question_ent_list = []
        for tuple in founded_tuple:
            if tuple[2] != question and tuple[1] not in question and tuple[1] not in question_ent_list:
                template_number +=1
                question_ent_list.append(tuple[1]) # do not keep questions with the same entity as the template
                line = train[tuple[-1]-1]
                new_fact = ""
                
                for num, r in enumerate(line["orig"]["new_triples_labeled"]):
                    if NatureL:
                        fact = " ".join([r[0], NL_dict[line["orig"]["new_triples"][num][1]], r[2]])+'.'
                    else:
                        fact = " ".join(r)+'.'
                    if num ==0:
                        new_fact = new_fact + fact
                    else:
                        new_fact = new_fact +'\n' + fact
                questions = random.choice(line['questions'])
                icl_examples += f'Question: {questions}\nAnswer: {new_fact}'+'\n\n'
            else:
                pass
            if template_number == args.template_number:
                break
    elif mode == 'ans':
        template_number = 0
        question_ent_list = []
        for tuple in founded_tuple:
            if tuple[2] != question and tuple[1] not in question and tuple[1] not in question_ent_list:
                template_number +=1
                question_ent_list.append(tuple[1]) # do not keep questions with the same entity as the template
                line = train[tuple[-1]-1]
                new_fact = "Given fact: "
                for num, r in enumerate(line["orig"]["new_triples_labeled"]):
                    if NatureL:
                        fact = " ".join([r[0], NL_dict[line["orig"]["new_triples"][num][1]], r[2]])+','
                    else:
                        fact = " ".join(r)+','
                    if num ==0:
                        new_fact = new_fact + fact
                    else:
                        new_fact = new_fact +' ' + fact
                questions = random.choice(line['questions'])
                target_new = line['new_answer']
                icl_examples += f'{new_fact} {questions}\nAnswer: {target_new}.'+'\n\n'
            else:
                pass
            if template_number == args.entropy_template_number:
                break
    
    return icl_examples

          
if __name__ == '__main__':
    
    args = parse_args()
    seed = args.seed
    set_seed(seed)
    
    if args.model =="gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(args.device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl") 
    elif args.model == 'vicuna':
        model = AutoModelForCausalLM.from_pretrained("/home/myid/xw54582/IIFT/weights/vicuna/hf_models/7B/", cache_dir="./cache").float().to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("/home/myid/xw54582/IIFT/weights/vicuna/hf_models/7B/", use_fast=False, padding_side="left", cache_dir="./cache")
        #template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: " #%s\nASSISTANT:"
    elif args.model == 'llama':
        model = AutoModelForCausalLM.from_pretrained("/home/myid/xw54582/IIFT/weights/llama/hf_models/7B/", cache_dir="./cache").float().to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("/home/myid/xw54582/IIFT/weights/llama/hf_models/7B/", use_fast=False, padding_side="left", cache_dir="./cache")
    elif args.model == 'llama-13b':
        model = AutoModelForCausalLM.from_pretrained("/home/myid/xw54582/IIFT/weights/vicuna/hf_models/13B/", cache_dir="./cache").float().to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("/home/myid/xw54582/IIFT/weights/vicuna/hf_models/13B/", use_fast=False, padding_side="left", cache_dir="./cache")
    elif args.model == 'llama2':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    elif args.model == 'falcon':
        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
    elif args.model == 't5-3b':
        model = AutoModelWithLMHead.from_pretrained('t5-3b').to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("t5-3b") 
    else:  
        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    print("Finished loading model")
    print("Model config", args)
    lines = load_dataset(f'data/{args.dataset}.json')
    train = load_dataset('data/MQuAKE-CF.json')
    edit_triplets_list = build_fact(lines)
    tuple_list = load_train_question("data/train_question_tuple.txt")


    relation_lines = load_dataset(args.relation_path)
    if args.NatureL:
        NL_dict = load_dataset('data/cloze_templates_NL.json')
        relation_dict, revserse_dict = build_relation(relation_lines, NL_dict)
    else:
        relation_dict, revserse_dict = build_relation(relation_lines, None)

    triplets_dict = load_triplets_dict(f'data/Wikidata_triplets_dict_{args.dataset}.pkl')
    orig_triplets_dict = load_triplets_dict('data/Wikidata_triplets_dict.pkl')
    print("Finished Loading")
    
    
    total_ques = 0
    raw_exact_match_cor = 0
    raw_par_match_cor = 0
    prun_exact_match_cor = 0
    prun_par_match_cor = 0
    pre_total = 0
    pre_exact_match = 0
    pre_par_match = 0
    total_raw_cor = 0
    total_prun_cor = 0
    
    
                    
    extractor = Extract(model, tokenizer, triplets_dict, relation_dict, revserse_dict, orig_triplets_dict, args)
    pruner = Prune(model, tokenizer, args)
    

    for i, line in enumerate(lines):
        
        if i <args.starting_line:
            pass
        else:

            print("\n\n")
            print(f"++++++++++++++++++++++++++++++++++{i+1}-th case++++++++++++++++++++++++++++++++++++++")
            
            total_ques +=1
            for j in range(3):
                
                print(f"+++++++++++++++++{j}-th question+++++++++++++++++++++")
                
                temp_raw_part_match_cor =0
                temp_raw_exact_match_cor =0
                temp_prun_part_match_cor =0
                temp_prun_exact_match_cor =0
                                
                temp_raw_cor = 0
                temp_prun_cor = 0
                
                if args.correctConflict:
                    original_fact_needed = []
                    total_triples = line["orig"]['new_triples']
                    edited_triples = line["orig"]['edit_triples']
                    should_not_edited_triples = [triple for triple in total_triples if triple not in edited_triples]
                    #print("should_not_edited_triples",should_not_edited_triples)
                    for should_not_edited_triple in should_not_edited_triples:
                        should_not_edit_hent_rel = (should_not_edited_triple[0], should_not_edited_triple[1])
                        if should_not_edit_hent_rel in edit_triplets_list:
                            print("Dataset Self-confliction detected!")
                            original_fact_needed.append(should_not_edit_hent_rel)
                            
                ground_truth_fact = set()
                new_triples = line["orig"]['new_triples']
                new_triples_labeled = line["orig"]['new_triples_labeled']
                for l, ll in enumerate(new_triples_labeled):
                    if args.NatureL:
                        fact_str = ' '.join([ll[0], NL_dict[new_triples[l][1]],ll[2]])
                    else:
                        fact_str = ' '.join(ll)
                    ground_truth_fact.add(fact_str)
            
                questions = line['questions'][j]
                print("Questions:", questions)
                print("Ground Truth:", ground_truth_fact)
                
                if args.template:
                    extract_prompt = tempalate_extractor(questions, args.NatureL, 'prob')
                    prom_questions = extract_prompt + 'Question: ' + questions + '\nAnswer:'
                    ans_prompt = tempalate_extractor(questions, args.NatureL, 'ans')  
                    #print("extract_prompt", extract_prompt)
                    #print("ans_prompt", ans_prompt)
                else:
                    prom_questions = questions
                
                #print("WHLOE PROMPT QUESTION:")
                #print(prom_questions)


                if args.mode == 'greedy':
                    print("wrong mode")
                elif args.mode == 'beam':
                    try:
                        raw_entity = ner_entity(questions)
                        try:
                            normlized_entity =  wikisearch.normalize(raw_entity)
                            print(f"Raw_entity:'{raw_entity}'. Normlized_entity:'{normlized_entity}'.")
                        except:
                            print(f"Raw_entity:'{raw_entity}'")
                        retrieved_fact_str = extractor.multi_hop_search(prom_questions, raw_entity, len(ground_truth_fact)+2, original_fact_needed)
                        pruned_fact_str = pruner.prune_fact(questions, retrieved_fact_str, ans_prompt)
                        print("=======Pruned Fact=======")
                        print(pruned_fact_str)
                        print('=========================')
                    except:
                        print(f"!!!!{i}-case-{j}-question:Fail to extract")
                        retrieved_fact_str = ''
                        pruned_fact_str = ''
                        
                    raw_fact_ans, raw_cor  = QA_func(model, tokenizer, line, retrieved_fact_str, questions, ans_prompt, 'Raw')
                    pruned_fact_ans, pruned_cor  = QA_func(model, tokenizer, line, pruned_fact_str, questions, ans_prompt, 'Pruned')
                    temp_raw_cor +=raw_cor
                    temp_prun_cor += pruned_cor
                    
                    
                    raw_part_match, raw_exact_match = match_func(ground_truth_fact, retrieved_fact_str, 'Raw')
                    prun_part_match, prun_exact_match = match_func(ground_truth_fact, pruned_fact_str, 'Pruned')

                    temp_raw_part_match_cor +=raw_part_match
                    temp_raw_exact_match_cor +=raw_exact_match
                    temp_prun_part_match_cor +=prun_part_match
                    temp_prun_exact_match_cor +=prun_exact_match
                                
                    if  temp_prun_cor > 0:
                        break 
                        
            if temp_raw_exact_match_cor > 0:
                    raw_exact_match_cor +=1
            if temp_raw_part_match_cor > 0:
                    raw_par_match_cor +=1
            if temp_prun_part_match_cor > 0:
                    prun_par_match_cor +=1
            if temp_prun_exact_match_cor > 0:
                    prun_exact_match_cor +=1
        
            if temp_raw_cor > 0:
                    total_raw_cor +=1
            if temp_prun_cor > 0:
                    total_prun_cor +=1

            print("++++++++++++++++++++++++++++++++++End++++++++++++++++++++++++++++++++++++++")
            
            if (i+1) % 10 == 0 and total_ques !=0:
                
                print(f"Finised on {i+1}-th case\n\
                raw_exact_match_acc: {raw_exact_match_cor/(total_ques)}, raw_par_match_acc: {raw_par_match_cor/(total_ques)}\n \
                prun_exact_match_acc: {prun_exact_match_cor/(total_ques)}, prun_par_match_acc: {prun_par_match_cor/(total_ques)}\n \
                raw_ans_acc: {total_raw_cor/(total_ques)}\n \
                prun_ans_acc: {total_prun_cor/(total_ques)}\n")  
                
            if (i+1) % 100 == 0: 
                
                print(f"!!Finised on {i+1}-th case, the 100-case prun_exact_match_acc: {(prun_exact_match_cor - pre_exact_match) /100}, prun_par_match_acc: {(prun_par_match_cor-pre_par_match)/100}")  
                pre_exact_match = prun_exact_match_cor
                pre_par_match = prun_par_match_cor
            
        
    print(f"Finised on {i+1}-th case, raw_exact_match_acc: {raw_exact_match_cor/(total_ques)}, raw_par_match_acc: {raw_par_match_cor/(total_ques)}\n \
            prun_exact_match_acc: {prun_exact_match_cor/(total_ques)}, prun_par_match_acc: {prun_par_match_cor/(total_ques)}\n \
            raw_ans_acc: {total_raw_cor/(total_ques)}\n \
            prun_ans_acc: {total_prun_cor/(total_ques)}\n")   