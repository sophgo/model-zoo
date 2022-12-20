import torch
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering
import json
import argparse
from torch import nn

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument(
        "--weight", default='./cvt_out.pt', type=str, required=False, help="Path to the weights."
    )
    parser.add_argument(
        "--save", default='./bert_traced.pt', type=str, required=False, help="Save path to where."
    )
    parser.add_argument(
        "--cfg", default='./bert_config.json', type=str, required=False, help="Bert model config."
    )
    parser.add_argument(
        "--weight_from_remote", action="store_true", help="load weight from huggingface model community."
    )
    parser.add_argument(
        "--load_pretrained", default='bert-large-uncased-whole-word-masking-finetuned-squad', type=str, required=False,
        help="Needed as long as weight_from_remote is set."
    )
    args = parser.parse_args()

    if not args.weight_from_remote:
        with open(args.cfg) as f:
            config_json = json.load(f)

        config = BertConfig(
            attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
            hidden_act=config_json["hidden_act"],
            hidden_dropout_prob=config_json["hidden_dropout_prob"],
            hidden_size=config_json["hidden_size"],
            initializer_range=config_json["initializer_range"],
            intermediate_size=config_json["intermediate_size"],
            max_position_embeddings=config_json["max_position_embeddings"],
            num_attention_heads=config_json["num_attention_heads"],
            num_hidden_layers=config_json["num_hidden_layers"],
            type_vocab_size=config_json["type_vocab_size"],
            vocab_size=config_json["vocab_size"])

        model = BertForQuestionAnswering(config)
        load_dict = torch.load(args.weight)
        model.classifier = model.qa_outputs
        model.load_state_dict(load_dict)
    else:
        model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", return_dict=False)

    model.eval()
    dummy_input = torch.zeros((1, 384), dtype=torch.long)
    # out = model(dummy_input, dummy_input, dummy_input)
    traced = torch.jit.trace(model, (dummy_input, dummy_input, dummy_input), strict=True)
    torch.jit.save(traced, args.save)
    print('----------- over -----------')

