import torch
import torch.nn as nn
from BD_data_load import idx2trigger, argument2idx, idx2argument
from BD_consts import NONE
from BD_utils import find_triggers
from transformers import (
    BertModel,
)
from utils import get_logger

logger = get_logger(name=__name__, log_file=None)

model_path = '/data2/models_nlp/pyt/chinese-macbert-base-model'


class Net(nn.Module):
    def __init__(self, trigger_size=None, argument_size=None, device=torch.device("cpu")):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = 768
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768, hidden_size=768 // 2, batch_first=True)
        self.linear_l = nn.Linear(hidden_size, hidden_size//2)
        self.linear_r = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size, argument_size),
        )
        self.device = device

    def predict_triggers(self, tokens_x, mask, head_indexes, arguments_true=None, Test=False):
        tokens_x = torch.LongTensor(tokens_x).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        head_indexes = torch.LongTensor(head_indexes).to(self.device)
        if self.training:
            self.bert.train()
            x, _ = self.bert(input_ids=tokens_x, attention_mask=mask, return_dict=False)
        else:
            self.bert.eval()
            with torch.no_grad():
                x, _ = self.bert(input_ids=tokens_x, attention_mask=mask, return_dict=False)
        batch_size = tokens_x.shape[0]
        # X shape: (bs, L, H)
        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes[i])
        trigger_logits = self.fc_trigger(x)
        # shape 2d
        triggers = trigger_logits.argmax(-1)

        x_rnn, h0, predicted_triggers_with_seq_num = [], [], []
        for i in range(batch_size):
            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in triggers[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor_l = self.linear_l(x[i, t_start, :])
                event_tensor_r = self.linear_r(x[i, t_end-1, :])
                event_tensor = torch.stack([event_tensor_l, event_tensor_r])
                h0.append(event_tensor)
                x_rnn.append(x[i])
                predicted_triggers_with_seq_num.append((i, t_start, t_end, t_type_str))

        argument_logits, arguments_true_match_predicted = [0], [0]
        # shape 2d
        arguments_predicted = [{'events': {}} for _ in range(batch_size)]
        if len(predicted_triggers_with_seq_num) > 0:
            h0 = torch.stack(h0, dim=1)
            c0 = torch.zeros(h0.shape, dtype=torch.float)
            c0 = c0.to(self.device)
            x_rnn = torch.stack(x_rnn)
            rnn_out, (hn, cn) = self.rnn(x_rnn, (h0,c0))
            argument_logits = self.fc_argument(rnn_out)
            # shape: (N_trigger L H), N is the number of predicted triggers
            argument_hat = argument_logits.argmax(-1)

            for i in range(len(argument_hat)):
                ba, st, ed, event_type_str = predicted_triggers_with_seq_num[i]
                if (st, ed, event_type_str) not in arguments_predicted[ba]['events']:
                    arguments_predicted[ba]['events'][(st, ed, event_type_str)] = []
                predicted_arguments = find_triggers([idx2argument[argument] for argument in argument_hat[i].tolist()])
                for predicted_argument in predicted_arguments:
                    e_start, e_end, e_type_str = predicted_argument
                    arguments_predicted[ba]['events'][(st, ed, event_type_str)].append((e_start, e_end, e_type_str))
            # shape 1d
            arguments_true_match_predicted = []
            if Test == False:
                # shape: (N_trigger L H), N is the number of predicted triggers
                for seq_num, t_start, t_end, t_type_str in predicted_triggers_with_seq_num:
                    a_label = [NONE] * x.shape[1]
                    if (t_start, t_end, t_type_str) in arguments_true[seq_num]['events']:
                        for (a_start, a_end, a_role) in arguments_true[i]['events'][(t_start, t_end, t_type_str)]:
                            for j in range(a_start, a_end):
                                if j == a_start:
                                    a_label[j] = 'B-{}'.format(a_role)
                                else:
                                    a_label[j] = 'I-{}'.format(a_role)
                    a_label = [argument2idx[t] for t in a_label]
                    arguments_true_match_predicted.append(a_label)

            arguments_true_match_predicted = torch.LongTensor(arguments_true_match_predicted).to(self.device)


        return trigger_logits, triggers, argument_logits, arguments_true_match_predicted, arguments_predicted
