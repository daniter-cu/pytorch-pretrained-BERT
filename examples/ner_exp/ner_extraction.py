import json, pickle
import torch

import sys
sys.path.append("../../../ner/BERT-NER/")

from bert import Ner

def get_ents(sent):
    try:
        output =  model.predict(sent)
    except:
        return None
    ents = []
    ent = None
    for k, v in output:
        assert k is not None, k
        if v['tag'] == 'O' and ent is not None:
            ents.append(ent.lower())
            ent = None
        if v['tag'].startswith('B'):
            if ent is not None:
                ents.append(ent.lower())
            ent = k
        if v['tag'].startswith('I') or v['tag'].startswith('X'):
            if ent is None:
                ent = k
            elif k.startswith("'"):
                ent += k
            else:
                ent += ' ' + k
    return ents


if __name__ == '__main__':

    model = Ner("../../../ner/BERT-NER/out/")
    device = torch.device("cuda")
    model.model.to(device)


    files = ["../../../Squad2Generative/data/train-v2.0.json", "../../../Squad2Generative/data/dev-v2.0.json"]
    for file in files:
        questions = []
        contexts = []
        examples = []
        labels = []
        id2idx = {}
        with open(file, 'r') as handle:
            jdata = json.load(handle)
            data = jdata['data']
        for i in range(len(data)):
            section = data[i]['paragraphs']
            for sec in section:
                context = sec['context']
                contexts.append(context)
                qas = sec['qas']
                for j in range(len(qas)):
                    question = qas[j]['question']
                    is_imp = qas[j]['is_impossible']
                    qid = qas[j]['id']
                    questions.append(question)
                    labels.append(is_imp)
                    examples.append((len(contexts) - 1, len(questions) - 1))
                    id2idx[qid] = len(questions) - 1
        data_type = file.split("/")[-1].split("-")[0]
        question_ents = []
        context_ents = []
        for q in questions:
            question_ents.append(get_ents(q))
            if len(question_ents) % 100 == 0:
                print("questions done:", len(question_ents))
        for c in contexts:
            context_ents.append(get_ents(c))
            if len(context_ents) % 100 == 0:
                print("Context ents done:", len(context_ents))
        with open(data_type+"-ents.pkl", "wb") as f:
            pickle.dump((context_ents, question_ents ) , f)

