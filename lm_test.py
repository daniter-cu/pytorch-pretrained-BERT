import torch
import json
import pickle
import numpy as np
import argparse

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def main(args):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = BertForMaskedLM.from_pretrained(args.model)
    model.eval()

    with open(args.squad, 'r') as handle:
        jdata = json.load(handle)
        data = jdata['data']

    def calc_prob(context, question):
        gt_question = question
        gt_q_tokens = tokenizer.tokenize(gt_question)
        gt_indexed_q_tokens = tokenizer.convert_tokens_to_ids(gt_q_tokens)

        # mask_tokens = ["[MASK]"]*len(gt_indexed_q_tokens)
        # indexed_mask_tokens = tokenizer.convert_tokens_to_ids(mask_tokens)

        context_tokens = tokenizer.tokenize(context)
        indexed_context_tokens = tokenizer.convert_tokens_to_ids(context_tokens)

        tokens_tensor = torch.tensor([indexed_context_tokens + gt_indexed_q_tokens])
        segments_tensors = torch.tensor([0]*len(indexed_context_tokens) + [1]*len(gt_indexed_q_tokens))
        predictions = model(tokens_tensor, segments_tensors)

        total = 0
        context_len = len(context_tokens)
        q_len = len(gt_indexed_q_tokens)
        for i in range(q_len):
            preds = predictions[0, context_len + i].data.numpy()
            total += -np.log(np.exp(preds[gt_indexed_q_tokens[i]]) / sum(np.exp(preds)))
        entropy = total / q_len
        perplexity = np.exp(entropy)
        return perplexity


    def calc_prob_nn(context, question):
        gt_question = question
        gt_q_tokens = tokenizer.tokenize(gt_question)
        gt_indexed_q_tokens = tokenizer.convert_tokens_to_ids(gt_q_tokens)

        context_tokens = tokenizer.tokenize(context)
        indexed_context_tokens = tokenizer.convert_tokens_to_ids(context_tokens)

        tokens_tensor = torch.tensor([indexed_context_tokens + gt_indexed_q_tokens])
        segments_tensors = torch.tensor([0] * len(indexed_context_tokens) + [1] * len(gt_indexed_q_tokens))
        predictions = model(tokens_tensor, segments_tensors)

        total = 0
        context_len = len(context_tokens)
        q_len = len(gt_indexed_q_tokens)
        for i in range(q_len):
            preds = predictions[0, context_len + i]
            m = torch.nn.LogSoftmax(0)
            total += -m(preds)[gt_indexed_q_tokens[i]]
        entropy = total / q_len
        perplexity = torch.exp(entropy)
        return perplexity.item()

    answerable_probs = []
    unanswerable_probs = []
    counter = 0
    for i in range(len(data)):
        section = data[i]['paragraphs']
        for sec in section:
            context = sec['context']
            qas = sec['qas']
            for j in range(len(qas)):
                question = qas[j]['question']
                label = qas[j]['is_impossible']
                try:
                    prob = calc_prob_nn(context, question)
                except:
                    continue
                if label:
                    unanswerable_probs.append(prob)
                else:
                    answerable_probs.append(prob)
                counter += 1
                if counter % 100 == 0:
                    print("Processed ", counter)
                if args.test_run and counter > 10:
                    with open("./%s.pkl" % args.output, "wb") as handle:
                        pickle.dump((answerable_probs, unanswerable_probs), handle)
                    return

    with open("./%s.pkl" % args.output, "wb") as handle:
        pickle.dump((answerable_probs, unanswerable_probs), handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True,
                        help="The bert model to load.")
    parser.add_argument("--output",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file to write the results to.")
    parser.add_argument("--squad",
                        default="../Squad2Generative/data/dev-v2.0.json",
                        type=str,
                        help="Name of squad file.")
    parser.add_argument("--test_run",
                        action='store_true',
                        default=False,
                        help="If true, shortcut the input data.")
    args = parser.parse_args()
    main(args)