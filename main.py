import torch
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm
import numpy as np
import logging

from utils_func import  *
from wiki_api.strings import question_token
from model import Extract, Prune 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def log_metrics(metrics, logger):
    total_ques = metrics['total_ques']
    logger.info(f"raw_exact_match_acc: {metrics['raw_exact_match_cor']/total_ques:.4f}")
    logger.info(f"raw_par_match_acc: {metrics['raw_par_match_cor']/total_ques:.4f}")
    logger.info(f"prun_exact_match_acc: {metrics['prun_exact_match_cor']/total_ques:.4f}")
    logger.info(f"prun_par_match_acc: {metrics['prun_par_match_cor']/total_ques:.4f}")
    logger.info(f"raw_ans_acc: {metrics['total_raw_cor']/total_ques:.4f}")
    logger.info(f"prun_ans_acc: {metrics['total_prun_cor']/total_ques:.4f}")

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
    
    MODEL_CONFIGS = {
    "gpt2": "gpt2-xl",
    "vicuna": "lmsys/vicuna-7b-v1.1",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "falcon": "tiiuae/falcon-7b",
    }
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIGS.get(args.model)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS.get(args.model))
    model.eval()
    logger.info("Finished loading model")
    logger.info(f"Model config: {args}")
    
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
    logger.info("Finished Loading")
    
    extractor = Extract(model, tokenizer, triplets_dict, relation_dict, revserse_dict, orig_triplets_dict, args)
    pruner = Prune(model, tokenizer, args)

    # Initialize evaluation metrics
    metrics = {
        'total_ques': 0,
        'raw_exact_match_cor': 0,
        'raw_par_match_cor': 0,
        'prun_exact_match_cor': 0,
        'prun_par_match_cor': 0,
        'total_raw_cor': 0,
        'total_prun_cor': 0
    }

    for i, line in enumerate(lines):
        if i < args.starting_line:
            continue

        logger.info(f"\n\n++++++++++++++++++++++++++++++++++{i+1}-th case++++++++++++++++++++++++++++++++++++++")
        
        metrics['total_ques'] += 1
        case_metrics = {k: 0 for k in metrics if k != 'total_ques'}

        for j in range(3):
            logger.info(f"+++++++++++++++++{j}-th question+++++++++++++++++++++")
            
            if args.correctConflict:
                original_fact_needed = []
                total_triples = line["orig"]['new_triples']
                edited_triples = line["orig"]['edit_triples']
                should_not_edited_triples = [triple for triple in total_triples if triple not in edited_triples]
                for should_not_edited_triple in should_not_edited_triples:
                    should_not_edit_hent_rel = (should_not_edited_triple[0], should_not_edited_triple[1])
                    if should_not_edit_hent_rel in edit_triplets_list:
                        logger.info("Dataset Self-confliction detected!")
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
            logger.info(f"Questions: {questions}")
            logger.info(f"Ground Truth: {ground_truth_fact}")
            
            if args.template:
                extract_prompt = tempalate_extractor(questions, args.NatureL, 'prob')
                prom_questions = extract_prompt + 'Question: ' + questions + '\nAnswer:'
                ans_prompt = tempalate_extractor(questions, args.NatureL, 'ans')  
            else:
                prom_questions = questions

            if args.mode == 'beam':
                try:
                    raw_entity = ner_entity(questions)
                    retrieved_fact_str = extractor.multi_hop_search(prom_questions, raw_entity, len(ground_truth_fact)+2, original_fact_needed)
                    pruned_fact_str = pruner.prune_fact(questions, retrieved_fact_str, ans_prompt)
                    logger.info("=======Pruned Fact=======")
                    logger.info(pruned_fact_str)
                    logger.info('=========================')
                except:
                    logger.error(f"!!!!{i}-case-{j}-question:Fail to extract")
                    retrieved_fact_str = ''
                    pruned_fact_str = ''
                    
                raw_fact_ans, raw_cor  = QA_func(model, tokenizer, line, retrieved_fact_str, questions, ans_prompt, 'Raw')
                pruned_fact_ans, pruned_cor  = QA_func(model, tokenizer, line, pruned_fact_str, questions, ans_prompt, 'Pruned')
                
                raw_part_match, raw_exact_match = match_func(ground_truth_fact, retrieved_fact_str, 'Raw')
                prun_part_match, prun_exact_match = match_func(ground_truth_fact, pruned_fact_str, 'Pruned')

                case_metrics['raw_par_match_cor'] += raw_part_match
                case_metrics['raw_exact_match_cor'] += raw_exact_match
                case_metrics['prun_par_match_cor'] += prun_part_match
                case_metrics['prun_exact_match_cor'] += prun_exact_match
                case_metrics['total_raw_cor'] += raw_cor
                case_metrics['total_prun_cor'] += pruned_cor

                if case_metrics['total_prun_cor'] > 0:
                    break

        # Update overall metrics
        for key in case_metrics:
            if case_metrics[key] > 0:
                metrics[key] += 1

        logger.info("++++++++++++++++++++++++++++++++++End++++++++++++++++++++++++++++++++++++++")
        
        if (i+1) % 10 == 0 and metrics['total_ques'] != 0:
            logger.info(f"Finished on {i+1}-th case")
            log_metrics(metrics, logger)

            
    logger.info(f"Finished on {i+1}-th case, final metrics:")
    log_metrics(metrics, logger)

