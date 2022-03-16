import torch.nn as nn
model_name = 'bert-base-uncased'
max_seq_len = 25


class BERTda(nn.Module):

    def __init__(self, bert):
        super(BERTda, self).__init__()
        self.bert_large = model_name == 'bert-large-uncased'
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 0 if bert_large is true, else skip it
        if self.bert_large:
            self.fc0 = nn.Linear(1024, 768)

        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        if self.bert_large:
            x = self.fc0(cls_hs)

            x = self.relu(x)

            x = self.dropout(x)
        else:
            x = cls_hs

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x
