from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score,precision_score, f1_score
import numpy as np


def evaluate_model(model,device,data_loader) :
  model.eval()
  losses = []
  all_predictions = []
  all_labels = []

  with torch.no_grad():
    for d in tqdm(data_loader) :
      sequences = d["tokens"].to(device).to(torch.long)
      targets = d["tags"].to(device).to(torch.long)
      paddings = d["masks"].to(device).to(torch.long)

      outputs = model(sequences, attention_mask=paddings, labels=targets)
      ids = torch.argmax(outputs[1],axis=2)

      flat_ids = ids.view(-1)
      flat_targets = targets.view(-1)
      masked_index = targets.view(-1) != -100
      unmasked_preds = torch.masked_select(flat_ids,masked_index)
      unmasked_tags = torch.masked_select(flat_targets,masked_index)
      all_predictions.extend(unmasked_preds.cpu())
      all_labels.extend(unmasked_tags.cpu())

      loss = outputs[0]
      losses.append(loss.item())

  accuracy = accuracy_score(all_predictions , all_labels)
  f_score = f1_score(all_predictions , all_labels, average = 'weighted',zero_division=1)

  return np.average(losses) , accuracy , f_score