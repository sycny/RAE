
import torch
from utils_func import find_lines_by_entity, retr_relations, sequences_prob, load_orig_entity, fact_triplet_to_sentence
from wiki_api.wikidata import id2entity, entity2id


class Extract:
    def __init__(self, model, tokenizer, triplets_dict, relation_dict, revserse_dict, orig_triplets_dict, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.triplets_dict = triplets_dict
        self.relation_dict = relation_dict
        self.revserse_dict = revserse_dict
        self.orig_triplets_dict = orig_triplets_dict

    def relation_prob(self, input_text, candidate_texts):
        #print("input_text", input_text)
        # Calculate probabilities for each candidate sequence
        candidate_probabilities = []
        if self.args.model.startswith('llama') or self.args.model.startswith('vicuna'):
            input_token = ["<s>"] + self.tokenizer.tokenize(input_text)
        else: 
            input_token = self.tokenizer.tokenize(input_text)
        with torch.no_grad():
            for cand_text in candidate_texts:
                if self.args.model.startswith('llama') or self.args.model.startswith('vicuna'): 
                    cand_token = self.tokenizer.tokenize(cand_text)
                else:
                    cand_token = self.tokenizer.tokenize(' ' + cand_text)
                prompt = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_token + cand_token)])
                probas = torch.softmax(self.model(prompt.to(self.model.device)).logits, -1).squeeze()
                cand_token_ids = self.tokenizer.convert_tokens_to_ids(cand_token)
                cand_token_prob = probas[torch.arange(len(input_token)-1, prompt.shape[1]-1), torch.tensor(cand_token_ids).long()]
                candidate_probabilities.append(torch.cumprod(cand_token_prob, 0).tolist()[-1] ** (1. / len(cand_token_ids)))
    
        return torch.tensor(candidate_probabilities).to(self.model.device)  
    
    def retr_fact_KG_sole_prob(self, input_text, text, entity, ent_eid, fact_needed):

            #here is a big bug, required fixing
            #eid = entity2id(entity) #get the eid e.g Q38
            eid_facts = find_lines_by_entity(self.triplets_dict, ent_eid)
            relation_names, _ = retr_relations(eid_facts, self.relation_dict)
            if relation_names == []:
                return None, None, None, None
            
            prob1 = self.relation_prob(text + '\n' + entity , relation_names)
            prob2 = self.relation_prob(text[len(input_text):] + '\n' + entity , relation_names)
             
            return_prob = prob1/prob2
            if self.args.loss == "prob_div":
                new_prob = (prob1/prob2)
            elif self.args.loss == "prob_div_log":
                new_prob = (prob1/prob2)*torch.log2(prob1/prob2)
                
            if new_prob.shape[0]>self.args.beam_width:
                key = self.args.beam_width
            else:
                key=new_prob.shape[0]
                
            top_prob, index = torch.topk(new_prob, key)
            return_top_prob = return_prob[index]
            selected_relations = [relation_names[i.item()] for i in index]
            #print('selected_relation', selected_relations)
            selected_pids = [self.revserse_dict[selected_relation] for selected_relation in selected_relations]
            
            next_rel_entities = []
        
            for selected_pid in selected_pids:
                for eid_fact in eid_facts:
                    eid, pid, tail_entites = eid_fact.split('\t')
                    if pid == selected_pid:
                        if self.args.correctConflict and (eid, pid) in fact_needed:
                            edited_tail_entites = tail_entites
                            tail_entites = load_orig_entity(self.orig_triplets_dict, eid, pid, tail_entites)
                            print(f"confilct-->original:{eid}\{pid}\{edited_tail_entites}-->{tail_entites}")
                        next_rel_entities.append(pid+'\t'+tail_entites)
                        break
            
            fact_set = [fact_triplet_to_sentence(f'{ent_eid}'+'\t'+ next_rel_entity, self.relation_dict) + '.' for next_rel_entity in next_rel_entities]
            entity_name = [id2entity(next_rel_entity.split('\t')[1]) for next_rel_entity in next_rel_entities]
            entity_eid = [next_rel_entity.split('\t')[1] for next_rel_entity in next_rel_entities]
            
            return (fact_set , entity_name, entity_eid, return_top_prob.to("cpu").tolist())
        
        
    def multi_hop_search(self, input_text, input_entity, iteration, fact_needed):
          
        for i in range(iteration):
            
            new_text_list = [] 
            next_entity_list = []
            next_entites_eid_list = []
            new_text_prob_list = []
            
            if i == 0:
                
                ent_eid = entity2id(input_entity)
                text_list = [input_text] 
                entity_list = [input_entity]
                entity_eid_list = [ent_eid]
                text_prob_list = [1.0]
                
            for j, text in enumerate(text_list):
                text_prob = text_prob_list[j]
                
                facts, next_entites, next_entites_eid, top_prob =  self.retr_fact_KG_sole_prob(input_text, text, entity_list[j], entity_eid_list[j], fact_needed)
                if facts is None:
                    continue
                for k, fact in enumerate(facts):
                    new_text = text + '\n' + fact
                    new_text_prob = top_prob[k]*text_prob
                    new_text_list.append(new_text)
                    next_entity_list.append(next_entites[k])
                    next_entites_eid_list.append(next_entites_eid[k])
                    new_text_prob_list.append(new_text_prob)
                    
            text_list = new_text_list
            #print(f"{i}-th text_list", [text[len(input_text):] for text in text_list])
            entity_list = next_entity_list
            entity_eid_list = next_entites_eid_list
            text_prob_list = new_text_prob_list
            #print(f"{i}-th text_prob_list", text_prob_list)
                    
        text_prob = torch.tensor(text_prob_list)
        
        if self.args.loss == "prob_div":
            final_score = text_prob
        elif self.args.loss == "prob_div_log":
            final_score = text_prob*torch.log2(text_prob)
        
        _, index = torch.topk(final_score, 1)
        
        
        fact = text_list[index][len(input_text + ' '):]
        print("======Final fact=====")
        print(fact)
            
        return fact     


class Prune:
    def __init__(self, model, tokenizer, args):
            self.model = model
            self.tokenizer = tokenizer
            self.args = args
            
    def ans_entroy(self, question, facts, entropy_prompt):
        
        input_text = ""
        for fact in facts:
            if input_text == "":
                input_text =  fact
            else:
                input_text =  input_text  + ', ' + fact 
        input_text = "Given fact: " + input_text + ', ' + question + '\nAnswer:'
        prom_text = entropy_prompt + input_text
        #print('prom_text', prom_text)
        
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prom_text, return_tensors="pt").to(self.args.device)
            beam0_logits = self.model(input_ids).logits[:, -1, :]
            beam0_prob = torch.softmax(beam0_logits, dim=-1)
            ans_h = -(beam0_prob * torch.log(beam0_prob)).sum()
        
        return ans_h 
    
    def facts_entropy(self, question, facts, entropy_prompt):
    
        num_inputs = len(facts)
        #print("num_inputs", num_inputs) 
        entropy_values = [0.0] * num_inputs
        
        test_fact = []
        
        for i, fact in enumerate(facts):
            test_fact.append(fact)
            ans_h = self.ans_entroy(question, test_fact, entropy_prompt)
            entropy_values[i] = ans_h
        
        entropy_values = torch.tensor(entropy_values)
        entropy_values = (entropy_values-torch.min(entropy_values))/(torch.max(entropy_values)-torch.min(entropy_values))
        
        return entropy_values.tolist()
    
    def prune_fact(self, question, facts_str, entropy_prompt):
        
        facts = facts_str.split('.\n')
        entropy_val = self.facts_entropy(question, facts, entropy_prompt)
        print("========================")
        print("entropy_val:", entropy_val)
        min_index = entropy_val.index(min(entropy_val))
        
        pruned_facts = facts[0:(min_index+1)] #need extra 1 to get the full chain
        pruned_facts_str = '.\n'.join(pruned_facts) + '.'
        
        return pruned_facts_str