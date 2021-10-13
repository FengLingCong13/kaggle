# author：FLC
# time:2021/10/13


import pandas as pd
import numpy as np
import torch
import transformers
import preprocess
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch


device = torch.device('cuda')


df_train, df_test, df_stopwords = preprocess.input_data()
df_train = preprocess.cortweetct_mislabeled_samples(df_train)
df_train['text'] = df_train['text'].apply(lambda s:preprocess.clean_data(s))
text_values = df_train['text'].values
labels = df_train['target'].values
tokenizer = transformers.BertTokenizer.from_pretrained('../bert-base-uncased')
print('Original Text : ', text_values[1])
print('Tokenized Text: ', tokenizer.tokenize(text_values[1]))
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[1])))

def encoder_fn(text_list):
    all_inputs_ids = []
    for text in text_list:
        input_ids = tokenizer.encode(text, add_special_tokens=True,
                                     max_length=160, padding = 'max_length', return_tensors='pt')
        all_inputs_ids.append(input_ids)
    all_inputs_ids = torch.cat(all_inputs_ids,dim=0)
    print(len(all_inputs_ids))
    return all_inputs_ids

all_inputs_ids = encoder_fn(text_values)
labels = torch.tensor(labels)

epochs = 60
batch_size = 32
dataset = TensorDataset(all_inputs_ids, labels)
train_size = int(0.90 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = transformers.BertForSequenceClassification.from_pretrained('../bert-base-uncased',
                                                                   num_labels=2, output_attentions=False,
                                                                   output_hidden_states=False)

model.to(device)
optimizer = transformers.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_data)*epochs
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
from sklearn.metrics import f1_score, accuracy_score, recall_score

def flat_accuary(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0,0
    total_eval_accuary = 0

    for step, batch in enumerate(train_data):
        model.zero_grad()
        # result = model(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0),
        #                      labels=batch[1])
        result = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
        loss = result.loss
        logits =result.logits
        total_loss += loss.item()
    total_loss += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
    optimizer.step()
    scheduler.step()

    model.eval()
    for i, batch in enumerate(val_data):
        with torch.no_grad():
            result = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
            loss = result.loss
            logits = result.logits
            total_val_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_eval_accuary += flat_accuary(logits, label_ids)

    avg_train_loss = total_loss / len(train_data)
    avg_val_loss = total_val_loss / len(val_data)
    avg_val_accuary = total_eval_accuary / len(val_data)

    print('Epochs:',epoch+1)
    print('Train loss:',round(avg_train_loss,6))
    print('Val loss:',round(avg_val_loss,6))
    print('Accuary:',round(avg_val_accuary,2)*100,'%')
    print('\n')


df_test['text'] = df_test['text'].apply(lambda s:preprocess.clean_data(s))
test_text_values = df_test['text'].values
test_all_inputs_ids = encoder_fn(test_text_values)
pred_data = TensorDataset(test_all_inputs_ids)
pred_dataloader = DataLoader(pred_data, batch_size=batch_size, shuffle=False)

model.eval()
preds = []
for i, (batch,) in enumerate(pred_dataloader):
    with torch.no_grad():
        result = model(batch.to(device), token_type_ids=None, attention_mask=(batch>0).to(device))
        logits = result.logits
        logits = logits.detach().cpu().numpy()
        preds.append(logits)

final_preds = np.concatenate(preds, axis=0)
final_preds = np.argmax(final_preds, axis=1)

submission = pd.DataFrame()
submission['id'] = df_test['id']
submission['target'] = final_preds
submission.to_csv('submission.csv',index=False)








