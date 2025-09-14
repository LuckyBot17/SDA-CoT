def get_implicit_relation_prompt(args, entities, extraction_explicit, sentence):
    prompt_prefix = "Given a sentence, all entities, and all explicit relationships within the sentence. Infer all possible implicit relationships between entities. For each pair of entities, infer up to "
    prompt_mid = " implicit relationships. Do NOT add any extra text, explanation, or punctuation! Every relationship stated as a triple: (Entity_name, Entity_name, Relation).\nSentence:"
    prompt_suffix = "\nRelation: "

    implicit_prompt = (prompt_prefix + args.infer_num + prompt_mid + sentence +"\n Explicit Relationships: "
                       + extraction_explicit + "\nEntities: " + entities +prompt_suffix)
    return implicit_prompt