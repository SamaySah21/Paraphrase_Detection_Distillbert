import transformers

MAX_LEN = 40
THRESHOLD_PROB = .84
EPOCHS = 10
DISTILED_BERT_VERSION = "distilbert-base-uncased"
POOLED_OUTPUT_DIM = 768
TOKENIZER = transformers.DistilBertTokenizer.from_pretrained(DISTILED_BERT_VERSION)
