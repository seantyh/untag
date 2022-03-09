import datasets
from datasets import Dataset
import pickle
from dataclasses import dataclass
from typing import Optional, Union
import torch
from transformers import BertTokenizerFast
import numpy as np
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer


def seeding(myseed):
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    np.random.seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    padding, trunc = True, True
    max_length =  None
    pad_to_multiple_of = None
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        pin_label = True if "label" in features[0].keys() else False
        # print("pinned label?", pin_label) # whether label column is correctly pinpointed
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"]) # 2
        flattened_features = [[{k: v[i] for k, v in feature.items()}
                            for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        batch = self.tokenizer.pad(
            flattened_features,
            padding= "max_length",
            max_length= self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
def main():
    #### parameters ####
    TESTSIZE = 0.2
    batch_size = 6
    numchoice = 2
    lr = 1e-5
    epochs = 20
    wd = 0.005
    myseed = 1027 # 43 # 1123
    gast = 1
    wmst = 20
    model_name = "RP-multiple_choice"
    datadir = "/home/avo727/data"
    tokenizer_name = 'bert-base-chinese'
    checkpoint_model = 'ckiplab/bert-base-chinese'
    seeding(myseed)
    print("* torch cuda archlist:", torch.cuda.get_arch_list())
    print("* torch cuda availability:", torch.cuda.is_available())
    print("* torch cuda device name:", torch.cuda.get_device_name(0))

    #### loading dataset ####
    datafile = f"{datadir}/rp_encoded_dataset.pkl"
    with open (datafile, 'rb') as F:
        encoded_dataset = pickle.load(F)

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    accepted_keys = ["input_ids", "attention_mask", "label"]
    features = [{k: v for k, v in encoded_dataset["train"][i].items() if k in accepted_keys} for i in range(10)]
    batch = DataCollatorForMultipleChoice()(features)
    check_batch = [tokenizer.decode(batch["input_ids"][8][i].tolist()) for i in range(numchoice)]
    # print(check_batch)
    #### loading model ####
    model = AutoModelForMultipleChoice.from_pretrained(checkpoint_model)
    args = TrainingArguments(
        model_name,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs= epochs,
        gradient_accumulation_steps = gast,
        weight_decay=wd,
        logging_strategy="epoch",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        warmup_steps = wmst, # about 500/(6*2) = 4x steps in one epoch, 4xx steps in total
    )
    #### training ####
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(),
        compute_metrics=compute_metrics,
    )
    TrainOutput = trainer.train()
    print(TrainOutput)

    ####### loading model for predictions ########

    # https://github.com/huggingface/transformers/issues/9398
    # "So I guess the trainer.predict() does really load the best model at the end of the training."
    print("Evaluating...")
    PredOutput = trainer.predict(
        test_dataset = encoded_dataset["test"]
    )
    labels = PredOutput.label_ids
    compute_metrics((PredOutput.predictions, labels))
    logits_path = f"{datadir}/RP_5thmodel_logits"
    preds_path = f"{datadir}/RP_5thmodel_predictions"
    np.save(logits_path, PredOutput.predictions)
    np.save(preds_path, preds)

if __name__ == '__main__':
    main()