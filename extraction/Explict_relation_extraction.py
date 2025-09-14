def get_explict_relation_prompt(entities, sentence):
    prompt_prefix = "Given a sentence, and all entities within the sentence. Extract all relationships between entities which directly stated in the sentence. Every relationship stated as a triple: (Entity_name, Entity_name, Relation). Do NOT add any extra text, explanation, or punctuation! \nSentence: "
    prompt_suffix = "\nRelaiton: "
    explict_relation_prompt = prompt_prefix + sentence + "\nEntities:" + entities + prompt_suffix
    return explict_relation_prompt