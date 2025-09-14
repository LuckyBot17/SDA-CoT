def get_ner_sentence(ner_prompt, sentence):
    complete_ner = ner_prompt + "\nsentence: " + sentence + "\nentities: "
    return complete_ner