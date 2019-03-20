from allennlp.predictors.predictor import Predictor
import json
import pickle
import spacy

def print_np(entry):
    phrase = ""
    for child in entry['children']:
        if child['nodeType'] != 'DT':
            phrase += child['word']+" "
    return(("["+entry['nodeType']+"]",phrase.strip()))

def get_q_parts(entry, nlp, tokens):
    if entry['nodeType'].startswith('VB'):
        if not nlp.vocab[entry['word'].lower()].is_stop:
            tokens.append(("["+entry['nodeType']+"]", entry['word']))
    elif entry['nodeType'].startswith("WH"):
        tokens.append(("["+entry['nodeType']+"]", entry['word']))
    elif entry['nodeType'] == 'NP':
        keep = True
        for child in entry['children']:
            if child['nodeType'] == 'NP' or child['nodeType'] == 'PP':
                keep = False
        if keep:
            tokens.append(print_np(entry)) # (entry['word'])
        else:
            if 'children' in entry and entry['children']:
                for child in entry['children']:
                    get_q_parts(child, nlp, tokens)
    else:
        if 'children' in entry and entry['children']:
            for child in entry['children']:
                get_q_parts(child, nlp, tokens)


def build_labels(dev_data_file, test_data_file, limit=None):
    nlp = spacy.load("en_core_web_sm")
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
    questions = {}

    for data_file in [dev_data_file, test_data_file]:
        with open(data_file, 'r') as handle: # update
            jdata = json.load(handle)
            data = jdata['data']
        for i in range(len(data)):
            section = data[i]['paragraphs']
            for sec in section:
                qas = sec['qas']
                for j in range(len(qas)):
                    qid = qas[j]['id']
                    question = qas[j]['question']
                    questions[qid] = question

    labels = {}
    counter = 0
    for id, q in questions.items() if limit is None else list(questions.items())[:limit]:
        res = predictor.predict(sentence=q)
        tokens = []
        get_q_parts(res['hierplane_tree']['root'], nlp, tokens)
        labels[id] = tokens
        counter += 1
        if counter % 1000 == 0:
            print("Finished with ", str(counter))

    with open("part_labels.pkl", "wb") as f:
        pickle.dump(labels, f)

if __name__ == '__main__':
    build_labels("../dataset/dev-v2.0.json", "../dataset/train-v2.0.json")