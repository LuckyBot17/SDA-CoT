def get_discrimination_prompt(args, entities, relation_inf, sent):
    prompt_prefix = '''Given a sentence, and all uncertain relationships within the sentence. Score the confidence level of each relationship. The confidence score ranges from 0 to 10, where a higher score indicates a higher likelihood of the relationship being correct.
Do NOT add any extra text, explanation, or punctuation! Every relationship stated as a triple: (Entity_name, Entity_name, Relation).\nSentence: '''
    prompt_suffix = '''\nScores: '''
    relation_prompt = (prompt_prefix + sent + "\nImplicit Relationships: " + relation_inf
                                + "\nEntities: " + entities + prompt_suffix)
    return relation_prompt