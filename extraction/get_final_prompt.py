def get_final_qa_prompt(args, entities, relation, sent):
    prompt_prefix = '''\nGiven a sentence, all entities and all relationships within the sentence. Answering the question. Every relationship stated as a triple: (E_A, E_B, Relation)\n '''

    prompt_suffix = '''\nPlease give the only answer choice as (A,B,C,D,E). \nAnswer: Let's think step by step.'''

    relation_prompt = (prompt_prefix + "\nEntities: " + entities
                       + "\nRelationships: " + relation
                       + "\nQuestion: " + sent + prompt_suffix)
    return relation_prompt