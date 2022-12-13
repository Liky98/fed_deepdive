from collections import OrderedDict
import warnings

import flwr as fl
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

import random
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric
from transformers import AdamW
#from transformers import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "roberta-base"


def load_data():
    dataset = load_dataset("silicone", "meld_e")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def encode(examples):
        return tokenizer(examples['Speaker'], examples['Utterance'], truncation=True, padding="max_length")

    population = random.sample(range(len(dataset["train"]['Idx'])), int(len(dataset["train"]['Idx'])/3))

    dataset["train"] = dataset["train"].select(population)
    dataset = dataset.map(encode, batched=True)
    #dataset["test"] = dataset["test"].select(population)
    dataset = dataset.remove_columns(['Utterance', 'Speaker', 'Emotion', 'Dialogue_ID', 'Utterance_ID','Idx'])

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "Label"])
    dataset = dataset.rename_column("Label", "labels")
   #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset.set_format("torch")
    train_dataloader = torch.utils.data.DataLoader(dataset['train'], shuffle=False, batch_size=2)
                                                   #collate_fn=data_collator)
    val_dataloader = torch.utils.data.DataLoader(dataset['validation'], shuffle=False, batch_size=2)
                                                 #collate_fn=data_collator)
    test_dataloader = torch.utils.data.DataLoader(dataset['test'], shuffle=False, batch_size=2)
                                                  #collate_fn=data_collator)

    return train_dataloader, val_dataloader, test_dataloader


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=1e-5)
    net.train()
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    prediction_list = []
    label_list = []

    #metric = load_metric("accuracy")
    #metric2= load_metric("f1")
    loss = 0
    net.eval()
    for batch in tqdm(testloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs[1]
        loss += outputs[0].item()
        predictions = torch.argmax(logits, dim=-1)
        prediction_list.extend(predictions.cpu())
        labels = batch['labels'].cpu()
        label_list.extend(labels.cpu())
        #metric.add_batch(predictions=predictions, references=batch["labels"])
        #metric2.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    #accuracy = metric.compute()["accuracy"]
    #f1 = metric2.compute(average="weighted")["f1"]

    f1 = f1_score(prediction_list, label_list, average='weighted')
    accuracy = accuracy_score(prediction_list, label_list)

    return loss, accuracy, f1


def main():
    global CHECKPOINT
    config = AutoConfig.from_pretrained(CHECKPOINT, num_labels=7)

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, config=config
    ).to(DEVICE)

    trainloader, val_loader, testloader = load_data()

    # Flower client
    class IMDBClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy, f1 = test(net, testloader)
            print(f'acc : {accuracy}, f1 : {f1}')
            return float(loss), len(testloader), {"accuracy": float(accuracy), "f1score": float(f1)}

    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=IMDBClient())


if __name__ == "__main__":
    main()
