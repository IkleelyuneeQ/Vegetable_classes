import torch
from torch import no_grad
from tqdm import tqdm


def train_model(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
    avg_loss = total_loss / total
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc


def eval_model(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
          outputs = model(images)
          loss = loss_fn(outputs, labels)

          total_loss += loss.item()
          total += labels.size(0)
          preds = torch.argmax(outputs, dim=1)
          correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

def predict(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return all_probs, all_preds, all_labels

