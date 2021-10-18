# author：FLC
# time:2021/10/13


import pandas as pd
import numpy as np
import transformers
import preprocess
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

device = torch.device('cuda')
# 导入数据
df_train, df_test, df_stopwords = preprocess.input_data()
# 纠正标记错误的样本
df_train = preprocess.cortweetct_mislabeled_samples(df_train)
# 进行数据清洗
# df_train['text'] = df_train['text'].apply(lambda s:preprocess.clean_data(s))
# 获得value值
text_values = df_train['text'].values
labels = df_train['target'].values
# 创建tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('../bert-base-uncased')
print('Original Text : ', text_values[1])  # 打印文本
print('Tokenized Text: ', tokenizer.tokenize(text_values[1]))  # 打印文本转换为token
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[1])))


# 对每一行句子进行填充
def encoder_fn(text_list):
    all_inputs_ids = []  # 用来存放每一行句子
    for text in text_list:  # 遍历每一行文本
        # add_special_tokens是添加首位标识符，return_tensors的pt是返回pytorch的tensor
        input_ids = tokenizer.encode(text, add_special_tokens=True,  # 进行padding
                                     max_length=160, padding='max_length', return_tensors='pt')
        all_inputs_ids.append(input_ids)
    all_inputs_ids = torch.cat(all_inputs_ids, dim=0)  # 进行concat起来
    return all_inputs_ids


all_inputs_ids = encoder_fn(text_values)  # 对输入进行padding操作
labels = torch.tensor(labels)  # labels转换为tensor

epochs = 4  # 定义迭代次数
batch_size = 32  # 定义batchsize
# TensorDataset是将两个Tensor进行打包，tensor的第一个维度要相同
dataset = TensorDataset(all_inputs_ids, labels)
# 计算训练集的大小
train_size = int(0.90 * len(dataset))
# 计算交叉验证集的大小
val_size = len(dataset) - train_size
# 进行随机划分
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# 定义数据加载器
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义模型，num_labels指定要分类的数目，num_labels指定分类的个数
model = transformers.BertForSequenceClassification.from_pretrained('../bert-base-uncased',
                                                                   num_labels=2, output_attentions=False,
                                                                   output_hidden_states=False)

model.to(device)
# 定义模型优化器
optimizer = transformers.AdamW(model.parameters(), lr=2e-5)
# 计算总的次数
total_steps = len(train_data) * epochs
# 定义学习率优化器
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 统计精度
def flat_accuary(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()  # 预测的值
    labels_flat = labels.flatten()  # 真实值
    return accuracy_score(labels_flat, pred_flat)  # 计算精度，第一个参数为正确值、第二个参数是预测值


for epoch in range(epochs):
    model.train()  # 模型训练
    total_loss, total_val_loss = 0, 0  # 定义损失
    total_eval_accuary = 0  # 模型精确率

    for step, batch in enumerate(train_data):  # 迭代器中取出batch
        model.zero_grad()  # 梯度清零
        # 执行前向传播
        result = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                       labels=batch[1].to(device))
        loss = result.loss  # 得到损失
        logits = result.logits  # 得到softmax之前的分类结果
        total_loss += loss.item()  # 统计总的损失
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 进行梯度裁剪，第二个参数是定义梯度的最大范数
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 学习率更新

    model.eval()  # 开启评价模式
    for i, batch in enumerate(val_data):  # 取出交叉验证集数据
        with torch.no_grad():  # 关闭梯度
            # 和上面一样，调用模型的forward函数
            result = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                           labels=batch[1].to(device))
            loss = result.loss  # 统计损失
            logits = result.logits  # 得到softmax之前的分类结果
            total_val_loss += loss.item()  # 统计损失
            logits = logits.detach().cpu().numpy()  # detach也就是返回没有梯度的tensor，numpy只能处理cpu的，因此要先转换为cpu
            label_ids = batch[1].to('cpu').numpy()  # 正确的标签
            total_eval_accuary += flat_accuary(logits, label_ids)  # 计算精度

    avg_train_loss = total_loss / len(train_data)  # 计算训练损失
    avg_val_loss = total_val_loss / len(val_data)  # 计算交叉验证集损失
    avg_val_accuary = total_eval_accuary / len(val_data)  # 计算精度

    print('Epochs:', epoch + 1)
    print('Train loss:', round(avg_train_loss, 6))
    print('Val loss:', round(avg_val_loss, 6))
    print('Accuary:', round(avg_val_accuary, 2) * 100, '%')
    print('\n')

df_test['text'] = df_test['text'].apply(lambda s: preprocess.clean_data(s))  # 对测试集进行数据清洗
test_text_values = df_test['text'].values
test_all_inputs_ids = encoder_fn(test_text_values)  # 对测试集进行padding
pred_data = TensorDataset(test_all_inputs_ids)  # 进行打包
pred_dataloader = DataLoader(pred_data, batch_size=batch_size, shuffle=False)  # 转换成迭代器

model.eval()  # 开启评估模式
preds = []  # 用于存放预测值
for i, (batch,) in enumerate(pred_dataloader):
    with torch.no_grad():  # 关闭梯度
        # 调用模型进行预测
        outputs = model(batch.to(device), token_type_ids=None, attention_mask=(batch > 0).to(device))
        # 取得输出的类别
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        # 加入预测值
        preds.append(logits)

final_preds = np.concatenate(preds, axis=0)
final_preds = np.argmax(final_preds, axis=1)  # 取得类别

submission = pd.DataFrame()
submission['id'] = df_test['id']   # 提交的id
submission['target'] = final_preds   # 提交的预测结果
submission.to_csv('submission.csv', index=False)



