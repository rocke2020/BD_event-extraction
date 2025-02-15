import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from BD_model import Net
from BD_data_load import TrainDataset, add_pad_for_train, TestDataset, Testpad, all_triggers, all_arguments
from BD_eval import eval
from BD_test import test
from utils import get_logger
from tqdm import tqdm

logger = get_logger(name=__name__, log_file=None)


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x, id, trigger_logits_true, arguments, seq_len, head_indexes, mask, words, triggers = batch
        optimizer.zero_grad()
        trigger_logits, triggers_predicted, argument_logits, arguments_true_match_predicted, arguments_predicted = (
            model.predict_triggers(
                tokens_x=tokens_x,
                mask=mask,
                head_indexes=head_indexes,
                arguments=arguments))
        trigger_logits_true = torch.LongTensor(trigger_logits_true).to(model.device)
        trigger_logits_true = trigger_logits_true.view(-1)
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, trigger_logits_true)

        if len(argument_logits) != 1:
            argument_logits = argument_logits.view(-1, argument_logits.shape[-1])
            argument_loss = criterion(argument_logits, arguments_true_match_predicted.view(-1))
            loss = trigger_loss + 2 * argument_loss
        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        if i % 40 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="output")
    parser.add_argument("--trainset", type=str, default="./data/train.json")
    parser.add_argument("--devset", type=str, default="./data/dev.json")
    parser.add_argument("--testset", type=str, default="./data/test1.json")

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'device: {device}')
    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        argument_size=len(all_arguments)
    )
    if device == 'cuda':
        model = model.cuda()

    train_dataset = TrainDataset(hp.trainset)
    dev_dataset = TrainDataset(hp.devset)
    test_dataset = TestDataset(hp.testset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=add_pad_for_train)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=add_pad_for_train)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=Testpad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    early_stop = 15
    stop = 0
    best_scores = 0.0
    for epoch in tqdm(range(1, hp.n_epochs + 1)):
        stop += 1
        print("=========train at epoch={}=========".format(epoch))
        train(model, train_iter, optimizer, criterion)

        fname = os.path.join(hp.logdir, str(epoch))
        print("=========dev at epoch={}=========".format(epoch))
        trigger_f1, argument_f1 = eval(model, dev_iter, fname + '_dev')

        # print("=========test at epoch={}=========".format(epoch))
        # test(model, test_iter, fname + '_test')
        if stop >= early_stop:
            print("The best result in epoch={}".format(epoch - early_stop - 1))
            break
        if trigger_f1 + argument_f1 > best_scores:
            best_scores = trigger_f1 + argument_f1
            stop = 0
            print("The new best in epoch={}".format(epoch))
            # torch.save(model, "best_model.pt")0
