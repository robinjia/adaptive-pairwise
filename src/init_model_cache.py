"""Make HuggingFace transformers download needed models to cache."""
from transformers import (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                          BertConfig, BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer, 'bert-base-uncased', True),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased', False),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base', False),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v2', False),
}

def main():
    for model in MODEL_CLASSES:
        config_class, model_class, tokenizer_class, name, do_lower_case = MODEL_CLASSES[model]
        config = config_class.from_pretrained(name, num_labels=2)
        tokenizer = tokenizer_class.from_pretrained(name, do_lower_case=do_lower_case)
        model = model_class.from_pretrained(name, config=config)

if __name__ == '__main__':
    main()
