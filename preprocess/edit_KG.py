from tqdm import tqdm
import pickle
import json

def load_dataset(path):
    
    with open(path, "r") as f:
        lines = json.load(f)
    
    return lines

def find_lines_by_entity(triplets_dict, entity_number):
    # Search for lines containing the given entity number
    if entity_number in triplets_dict:
        return triplets_dict[entity_number]
    return []

def save_triplets_dict(triplets_dict, filename):
    with open(filename, 'wb') as file:
        pickle.dump(triplets_dict, file)

def load_triplets_dict(filename):
    with open(filename, 'rb') as file:
        triplets_dict = pickle.load(file)
    return triplets_dict

def main():
    filename = '/home/myid/code/ILME/edit/data/wikidata5m_all_triplet.txt'  # Replace this with your file path
    #counterfact_path = '/home/myid/code/ILME/edit/data/MQuAKE-CF-3k.json'
    counterfact_path = '/home/myid/code/ILME/edit/data/MQuAKE-T.json'
    dict_filename = '/home/myid/code/ILME/edit/data/Wikidata_triplets_dict.pkl'  # Replace this with the desired path for saving the dictionary
    #new_dict_filename = '/home/myid/code/ILME/edit/data/Wikidata_triplets_dict_Edited_CF_3k.pkl' 
    new_dict_filename = '/home/myid/code/ILME/edit/data/Wikidata_triplets_dict_Edited_T.pkl' 
    # Step 1: Load triplets from the file into a dictionary
    triplets_dict = load_triplets_dict(dict_filename)
    lines = load_dataset(counterfact_path)
    edit_fact_list = []
    for line in lines:
        
        #for edit_fact in line["orig"]["edit_triples"]:
        for edit_fact in line["orig"]["edit_triples"]:
            
            head_e, rel, tail_e = edit_fact
            edit_fact_list.append((head_e, rel))
            lines_containing_entity = find_lines_by_entity(triplets_dict, head_e)
            
            if len(lines_containing_entity) ==0:
                triplets_dict[head_e] = [f"{head_ent}\t{relation}\t{tail_e}"]
            else:
                new_lines_containing_entity = [fact for fact in lines_containing_entity]
                for i, existing_line in enumerate(lines_containing_entity):
                    head_ent, relation, _ = existing_line.split('\t')
                    if rel == relation:
                        new_lines_containing_entity.remove(existing_line) ##remove all the facts with this relatino
                new_lines_containing_entity.append(f"{head_ent}\t{rel}\t{tail_e}") # only keep the edited verison
                triplets_dict[head_e] = new_lines_containing_entity
                
        for edit_fact in line["orig"]["new_triples"]:
            head_e, rel, tail_e = edit_fact
            if (head_e, rel) in edit_fact_list: # if the fact appears in the edited fact, pass
                pass
            else:
                lines_containing_entity = find_lines_by_entity(triplets_dict, head_e)
                if len(lines_containing_entity) ==0:     # if the head entitiy has no facts, append this one
                    triplets_dict[head_e] = [f"{head_ent}\t{relation}\t{tail_e}"]
                else:
                    new_lines_containing_entity = [fact for fact in lines_containing_entity]
                    for i, existing_line in enumerate(lines_containing_entity):
                        head_ent, relation, _ = existing_line.split('\t')
                        if rel == relation:
                            new_lines_containing_entity.remove(existing_line)
                    new_lines_containing_entity.append(f"{head_ent}\t{rel}\t{tail_e}")
                    triplets_dict[head_e] = new_lines_containing_entity

    save_triplets_dict(triplets_dict, new_dict_filename)
                    
if __name__ == "__main__":
    main()
