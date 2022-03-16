import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast

from da_model import BERTda, model_name, max_seq_len
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = AutoModel.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

if model_name == "bert-base-uncased":
    path = 'checkpoints/saved_weights.pt'
else:
    path = 'checkpoints/large_saved_weights_3.1.pt'

model = BERTda(bert)
model.load_state_dict(torch.load(path))
model = model.to(device)


def predict(lines: list):
    """
    Predict the if there is an actionable item in the sentences.
    :param lines:
    :return:
    """
    lines = [line.strip().lower() for line in lines]
    tokens_test = tokenizer.batch_encode_plus(
        lines,
        max_length=max_seq_len,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        return preds


if __name__ == '__main__':
    df_test = pd.read_csv("data/test.csv", nrows=2000)
    print(df_test.head())
    print(df_test.shape)
    print(df_test['label'].value_counts(normalize=True))

    preds = predict(df_test['text'].tolist())
    test_y = torch.tensor(df_test['label'].tolist())
    print(classification_report(test_y, preds))

