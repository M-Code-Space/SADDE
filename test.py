import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, average_precision_score, \
    confusion_matrix

def test_model(model, test_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
    accuracy = 100 * correct / total
    print(f"Model Accuracy: {accuracy:.2f}%")