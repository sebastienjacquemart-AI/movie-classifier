import evaluate 
import numpy as np

from transformers import AutoModelForSequenceClassification, TrainingArguments

import trainer.boiler as boiler
import trainer.data as data

def get_training_args():
    training_args = TrainingArguments(
        output_dir="check_deployed_model",
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    return training_args

def compute_metrics(eval_pred):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    predictions, labels = eval_pred
    predictions = boiler.sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))

def get_solution(classes, model_path="microsoft/deberta-v3-small"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=len(classes),
        id2label=data.get_id2class(classes),
        label2id=data.get_class2id(classes),
        problem_type = "multi_label_classification"
    )

    return model


