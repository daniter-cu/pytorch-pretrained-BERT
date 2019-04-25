import random
import json
import pickle
import spacy
import sys
from allennlp.predictors.predictor import Predictor
from pytorch_pretrained_bert.tokenization import BertTokenizer



predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
nlp = spacy.load("en_core_web_sm")
strip_stop = ["a", "an", "and", "as", "or", "the", "that", "which", "when", "whose", "is", "was", "what", "to", "of"]


def individual_filter(term):
    if term.startswith("##"):
        return False
    if term in nlp.vocab and (nlp.vocab[term].is_stop or nlp.vocab[term].is_punct):
        return False
    return True


def simple_filter(term):
    if term in strip_stop or (term in nlp.vocab and nlp.vocab[term].is_punct):
        return False
    return True


def strip_terms(phrase):
    fltr = [simple_filter(t) for t in phrase]
    start = fltr.index(True)
    end = list(reversed(fltr)).index(True)
    return phrase[start:len(fltr) - end]


def matcher(source, target):
    i = 0
    matches = []
    while i < len(source):
        ii = 1
        current = []
        for j in range(len(target)):
            if source[i] == target[j]:
                cand = []
                for ii in range(len(source) - i):
                    if j + ii > len(target) - 1:
                        break
                    if source[i + ii] == target[j + ii]:
                        cand.append(source[i + ii])
                    else:
                        if len(cand) > len(current):
                            current = list(cand)
                        break
                if len(cand) > len(current):
                    current = list(cand)
        if current:
            matches.append(current)
        i += len(current) if current else 1

    ## filters
    matches = [m for m in matches if sum(
        [individual_filter(token) for token in m]) > 0]

    matches = [strip_terms(m) for m in matches]
    return matches


def gather_spans(o, out):
    out[o['word']] = o['nodeType']
    if 'children' in o:
        for child in o['children']:
            gather_spans(child, out)


def x_in_y(query, base):
    try:
        l = len(query)
    except TypeError:
        l = 1
        query = type(base)((query,))

    for i in range(len(base)):
        if base[i:i + l] == query:
            return True
    return False


def get_copies(q, c, parse):
    copies = matcher(tokenizer.tokenize(q), tokenizer.tokenize(c))
    spans = {}
    gather_spans(parse, spans)
    annotated_copies = []
    for s in copies:
        min_span = (None, None)
        for cand in spans.keys():
            if x_in_y(s, tokenizer.tokenize(cand)) and (min_span[0] is None or len(cand) < len(min_span[0])):
                min_span = (cand, spans[cand])
        annotated_copies.append((s, min_span))
    return annotated_copies


def print_np(entry):
    phrase = ""
    for child in entry['children']:
        if child['nodeType'] != 'DT':
            phrase += child['word'] + " "
    return (("[" + entry['nodeType'] + "]", phrase.strip()))


def get_q_parts(entry, nlp, tokens):
    if entry['nodeType'].startswith('VB'):
        if not nlp.vocab[entry['word'].lower()].is_stop:
            tokens.append(("[" + entry['nodeType'] + "]", entry['word']))
    elif entry['nodeType'].startswith("WH"):
        tokens.append(("[" + entry['nodeType'] + "]", entry['word']))
    elif entry['nodeType'] == 'NP':
        keep = True
        for child in entry['children']:
            if child['nodeType'] == 'NP' or child['nodeType'] == 'PP':
                keep = False
        if keep:
            tokens.append(print_np(entry))  # (entry['word'])
        else:
            if 'children' in entry and entry['children']:
                for child in entry['children']:
                    get_q_parts(child, nlp, tokens)
    else:
        if 'children' in entry and entry['children']:
            for child in entry['children']:
                get_q_parts(child, nlp, tokens)


def get_q_parts_tmp(entry, nlp, tokens, siblings=None):
    if entry['nodeType'].startswith('VB'):
        prepend = ""
        rb_found = False
        if siblings:
            for sib in siblings:
                if sib['nodeType'] == 'RB':
                    rb_found = True
                if sib['nodeType'] in ['VBZ', 'RB', 'MD', 'JJ', 'VBD']:
                    prepend += sib['word'] + " "
        if not rb_found:
            prepend = ""
        if not nlp.vocab[entry['word'].lower()].is_stop:
            tokens.append(("[" + entry['nodeType'] + "]", prepend + entry['word']))
    elif entry['nodeType'].startswith("WH"):
        tokens.append(("[" + entry['nodeType'] + "]", entry['word']))
    elif entry['nodeType'] == 'NP':
        keep = True
        for child in entry['children']:
            if child['nodeType'] == 'NP' or child['nodeType'] == 'PP':
                keep = False
        if keep:
            nodeType, word = print_np(entry)
            prepend = ""
            rb_found = False

            if siblings:
                for sib in siblings:
                    if sib['nodeType'] == 'RB':
                        rb_found = True
                    if sib['nodeType'] in ['VBZ', 'RB', 'MD', 'JJ', 'VBD']:
                        prepend += sib['word'] + " "
            if not rb_found:
                prepend = ""
            tokens.append((nodeType, prepend + word))  # (entry['word'])
        else:
            if 'children' in entry and entry['children']:
                for child in entry['children']:
                    get_q_parts(child, nlp, tokens)
    else:
        if 'children' in entry and entry['children']:
            #             has_rb = False
            #             for child in entry['children']:
            #                 if child['nodeType'] == 'RB':
            #                     has_rb = True
            #             if has_rb:
            #                 for child in entry['children']:
            #                     print(child['nodeType'], child['word'])

            siblings = [] if siblings is None else siblings.copy()
            siblings.extend(entry['children'])
            for child in entry['children']:
                get_q_parts_tmp(child, nlp, tokens, siblings)


def get_const_chunks(q):
    res = predictor.predict(sentence=q)
    tokens = []
    get_q_parts_tmp(res['hierplane_tree']['root'], nlp, tokens)
    return tokens, res['hierplane_tree']['root']


def get_token_span(tokens, q):
    #     start = q.find(tokens[0])
    #     end_tok = tokens[-1] if not tokens[-1].startswith("##") else tokens[-1][2:]
    #     end = q.find(end_tok) + len(end_tok)
    #     return q[start:end]
    s = ""
    for t in tokens:
        if s == "":
            s += t
        elif t.startswith("##"):
            s += t[2:]
        elif t in nlp.vocab and nlp.vocab[t].is_punct:
            s += t
        else:
            s += " " + t
    return s


def get_overlap(tokens, span, part, copied_start, copied_end, const_part, const_span, const_start, const_end, q_tokens):
    if (copied_start < const_start and copied_end > const_start) or (
            const_start < copied_start and const_end > copied_start):
        min_start = min(const_start, copied_start)
        max_end = max(const_end, copied_end)
        ret_span = q_tokens[min_start:max_end]
        #         if min_start == copied_start and max_end == copied_end:
        #             ret_span = get_token_span(tokens, q)
        ret_part = "SPTK"
        if min_start == const_start and max_end == const_end:
            ret_part = const_part
        if min_start == copied_start and max_end == copied_end:
            ret_part = part
        return (ret_span, (None, ret_part))

    else:
        return None


def x_in_y_int(query, base):
    try:
        l = len(query)
    except TypeError:
        l = 1
        query = type(base)((query,))

    for i in range(len(base)):
        if base[i:i + l] == query:
            return i
    return -1


def combine_chunks(copied_chunks, q_chunks, parse, q):
    output_chunks = []
    overlap_marker = [False] * len(q_chunks)
    q_tokens = tokenizer.tokenize(q)
    for tokens, (span, part) in copied_chunks:
        copied_start = x_in_y_int(tokens, q_tokens)
        copied_end = copied_start + len(tokens)
        found = False
        for i, (const_tokens, (const_span, const_part)) in enumerate(q_chunks):
            const_start = x_in_y_int(const_tokens, q_tokens)
            const_end = const_start + len(const_tokens)
            overlap = get_overlap(tokens, span, part, copied_start, copied_end,
                                  const_part, const_span, const_start, const_end, q_tokens)
            if overlap:
                output_chunks.append(overlap)
                added = True
                found = True
                overlap_marker[i] = True
                break

        if found == False:
            output_chunks.append((tokens, (span, part)))
    start_ends = []
    for token_span, _ in output_chunks:
        start = x_in_y_int(token_span, q_tokens)
        start_ends.append((start, start + len(token_span)))

    for i, (const_tokens, (const_span, const_part)) in enumerate(q_chunks):
        if overlap_marker[i] == False:
            const_start = x_in_y_int(const_tokens, q_tokens)
            const_end = const_start + len(const_tokens)
            found = False
            for start, end in start_ends:
                if const_start >= start and const_end <= end:
                    found = True
            if not found:
                output_chunks.append((const_tokens, (const_span, const_part)))

    for i, (itokens, (ispan, ipart)) in enumerate(output_chunks):
        for j, (jtokens, (jspan, jpart)) in enumerate(output_chunks):
            if i == j:
                continue
            i_start = x_in_y_int(itokens, q_tokens)
            i_end = i_start + len(itokens)
            j_start = x_in_y_int(jtokens, q_tokens)
            j_end = j_start + len(jtokens)
            if (i_start <= j_start and i_end > j_start) or (j_start <= i_start and j_end > i_start):
                if len(itokens) > len(jtokens):
                    output_chunks.remove((jtokens, (jspan, jpart)))
                else:
                    output_chunks.remove((itokens, (ispan, ipart)))
                break

    return output_chunks


def tokenize_q_chunks(q_chunks):
    return [(tokenizer.tokenize(span), (span, part)) for part, span in q_chunks]


def order_chunks(chunks, parse):
    chunks_cp = list(chunks)
    questions = []
    other = []
    for chunk in chunks_cp:
        tokens, (span, part) = chunk
        if part.strip("[]").startswith("W"):
            questions.append(chunk)
        else:
            other.append(chunk)
    questions.sort(key=lambda x: len(x[0]), reverse=True)
    other.sort(key=lambda x: len(x[0]), reverse=True)
    return other + questions


def get_ordered_qsegs(q, c):
    q_chunks, parse = get_const_chunks(q)
    q_chunks = tokenize_q_chunks(q_chunks)
    copied_chunks = get_copies(q, c, parse)

    combined_chunks = combine_chunks(copied_chunks, q_chunks, parse, q)
    #print("#" * 20)
    ordered_qsegs = order_chunks(combined_chunks, parse)

    # clean up pparts
    ordered_qsegs = [(tokens, (span, part.strip("[]"))) for (tokens, (span, part)) in ordered_qsegs]

    return ordered_qsegs

def build_labels(dev_data_file, test_data_file, limit=None):
    questions = {}
    q2context = {}
    contexts = []
    for data_file in [dev_data_file, test_data_file]:
        with open(data_file, 'r') as handle: # update
            jdata = json.load(handle)
            data = jdata['data']
        for i in range(len(data)):
            section = data[i]['paragraphs']
            for sec in section:
                qas = sec['qas']
                context = sec['context']
                contexts.append(context)
                for j in range(len(qas)):
                    qid = qas[j]['id']
                    question = qas[j]['question']
                    questions[qid] = question
                    q2context[qid] = len(contexts) - 1

    labels = {}
    counter = 0
    section = int(sys.argv[1])
    chunk = 6000
    for id, q in list(questions.items())[chunk*section:chunk*(section+1)] if limit is None else list(questions.items())[:limit]:
        c = contexts[q2context[id]]
        chunks = get_ordered_qsegs(q, c)
        labels[id] = chunks
        counter += 1
        if counter % 1000 == 0:
            print("Finished with ", str(counter))

    with open("parsed_qs_labels"+str(section)+".pkl", "wb") as f:
        pickle.dump(labels, f)

def test_labels(dev_data_file, test_data_file):
    questions = {}
    q2context = {}
    contexts = []
    for data_file in [dev_data_file, test_data_file]:
        with open(data_file, 'r') as handle: # update
            jdata = json.load(handle)
            data = jdata['data']
        for i in range(len(data)):
            section = data[i]['paragraphs']
            for sec in section:
                qas = sec['qas']
                context = sec['context']
                contexts.append(context)
                for j in range(len(qas)):
                    qid = qas[j]['id']
                    question = qas[j]['question']
                    questions[qid] = question
                    q2context[qid] = len(contexts) - 1

    for i in random.sample(range(len(questions)), 1):
        print(i)
        ci = q2context[list(questions.keys())[i]]
        q = list(questions.values())[i]
        c = contexts[ci]
        print(q)
        print(get_ordered_qsegs(q, c))
    print(len(questions))


if __name__ == '__main__':
    build_labels("../../Squad2Generative/data/dev-v2.0.json", "../../Squad2Generative/data/train-v2.0.json")
