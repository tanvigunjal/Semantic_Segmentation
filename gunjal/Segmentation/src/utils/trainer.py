import torch 
import tqdm

from evaluation_metrics import *

def train(train_loader, model, optimizer, criterion, device, config, epoch_i):
        count = 0
        model.train()
        
        losses = []
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            # model prediction
            pred = model(images)

            # loss calculation
            loss = criterion(pred, labels)
            losses.append(loss.item())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if count % 100 == 0:
                print(f'Epoch: {epoch_i}/{config.epoch_total}, Iteration: {i}/{len(train_loader)}, Loss: {loss.item()}')

            count += 1

        return losses


def validation(val_loader, model, epoch_i, criterion, device, config):
    model.eval()

    # enable calculation of confusion matrix for n_classes = 19
    metrics = runningScore(19)
    accuracy = []
    jaccard = []

    with torch.no_grad():
        for i, (val_images, val_labels) in tqdm(enumerate(val_loader)):

            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            
            # model prediction
            val_pred = model(val_images)

            # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
            # considering predictions with highest scores for each pixel among 19 classes
            prediction = val_pred.data.max(1)[1].cpu().numpy()
            ground_truth = val_labels.data.cpu().numpy()

            # updation Metric
            metrics.update(prediction, ground_truth)
            mat = get_metrics(ground_truth.flatten(), prediction.flatten())
            accuracy.append(mat[0])
            jaccard.append(mat[1])

    score = metrics.get_scores()
    metrics.reset()

    accuracy_score = sum(accuracy) / len(accuracy)
    jaccard_score = sum(jaccard) / len(jaccard)
    score['accuracy'] = accuracy_score
    score['jaccard'] = jaccard_score

    print(f'Epoch: {epoch_i}/{config.epoch_total}, Accuracy: {accuracy_score}, Jaccard: {jaccard_score}')

    return score        