import pandas as pd
import numpy as np
import json
import re
from __future__ import unicode_literals, print_function
import plac
import re
import random
from pathlib import Path
import spacy
import json




df = pd.read_excel("quora_questions_for_entity.xlsx",sheet_name="Sheet1")
entity_data = {}


ANATOMY = df["ANATOMY"].values.tolist()
ANATOMY.sort(key = lambda s: len(str(s)))
ANATOMY = list(reversed(ANATOMY))
entity_data.update({"ANATOMY":ANATOMY})


DEMOGRAPHIC = df["DEMOGRAPHIC"].values.tolist()
DEMOGRAPHIC.sort(key = lambda s: len(str(s)))
DEMOGRAPHIC = list(reversed(DEMOGRAPHIC))
entity_data.update({"DEMOGRAPHIC":DEMOGRAPHIC})

DEVICES = df["DEVICES"].values.tolist()
DEVICES.sort(key = lambda s: len(str(s)))
DEVICES = list(reversed(DEVICES))
entity_data.update({"DEVICES":DEVICES})

DRUGS = df["DRUGS"].values.tolist()
DRUGS.sort(key = lambda s: len(str(s)))
DRUGS = list(reversed(DRUGS))
entity_data.update({"DRUGS":DRUGS})

FINDINGS = df["FINDINGS"].values.tolist() 
FINDINGS.sort(key = lambda s: len(str(s)))
FINDINGS = list(reversed(FINDINGS))
entity_data.update({"FINDINGS":FINDINGS})

PROBLEMS = df["PROBLEMS"].values.tolist()
PROBLEMS.sort(key = lambda s: len(str(s)))
PROBLEMS = list(reversed(PROBLEMS))
entity_data.update({"PROBLEMS":PROBLEMS})

PROCEDURES = df["PROCEDURES"].values.tolist()
PROCEDURES.sort(key = lambda s: len(str(s)))
PROCEDURES = list(reversed(PROCEDURES))
entity_data.update({"PROCEDURES":PROCEDURES})
#"led" "co" "mct" "aim" "thc" "hla" "atp" "msm" "uk"



stop_words = set(["best", "way", "much", "many", "same", "time", "easy", "ways", "i", "m", "i'", "isn't", "should", "could", "would", "shall", "can", "will", "",
"shouldn't", "couldn't", "wouldn't", "shalln't", "can't", "will", "won't", "not", "first", "last", "what", "where", "how", "who", "there", "this",
"that", "it", "they", "those", "these", "them", "was", "is", "am", "are", "do", "done", "did","my","name","is","his","her","afraid","ideas","ideal","ill",
"effect","iit","eat","induced","day","home","fasting","years","days","hours","year","day","hour","month","months","here","there","it","here","there","much",
"many","few","i","me","we","he","she","it","affect","other","associated","point","egg","his","her","he","she","wake up","air","hands feet","skip","go","went","gone","pick","treat","treatment"])



df2 = pd.read_excel("diab.xlsx",sheet_name="Sheet1")

questions = df2.loc[:,0].values.tolist()
questions = list(map(lambda x: str(x).lower(),questions))
questions = list(map(lambda x: re.sub(r"\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\_|\+|\=|\{|\}|\||\:|\"|\<|\>|\?|\[|\]|\\|\;|\'|\,|\/|\."," ",str(x)),questions))
questions = list(map(lambda x:re.sub(r"\s{2,}"," ",x),questions))



def f(entity_list,sentence,item,entity_name):
    a = re.split(r'\s'+item+r'\s',sentence)
    if len(a) < 3:
        start_index = len(a[0])
        end_index = start_index + len(item)
        entity_list.append((start_index,end_index,entity_name,item))
    else:
#         print("-------------begin----------------------")
        for index,value in enumerate(a[0:len(a)-1]):
            start_index = len(a[index])
            temp = index
            while temp-1 >=0 :
                temp = temp - 1
                start_index =  start_index + len(a[temp])+len(" "+item+" ")
            end_index = start_index + len(item)
            
            entity_list.append((start_index,end_index,entity_name,item))
    return entity_list

preference_order = ["PROBLEMS","PROCEDURES","DRUGS","DEVICES","FINDINGS","ANATOMY","DEVICES"]


train_data = []
count = 0
for question in questions:
    question = " "+str(question)+" "
    entity_list = []
    for entity_ in preference_order:
        for item in entity_data[entity_]:
            if type(item) is float or type(item) is int:
                pass
            else:
                if " "+item+" " in question:
                    if re.search(r"\w+",item):
                        if item not in stop_words:
                            entity_list = f(entity_list,question,item,entity_)
    train_data.append((question.strip(),{"entities":entity_list}))



TRAIN_DATA = train_data
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model="spacyModel", output_dir="spacyModel", n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    #     print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        # print("Loading from", output_dir)
        # nlp2 = spacy.load(output_dir)
        # for text in TEST_DATA:
        #     doc = nlp2(text)
        #     print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            # print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)