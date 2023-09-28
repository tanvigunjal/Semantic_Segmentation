import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

from model.unet import U_Net_model

def test(val_loader, model, criterion, epoch_i, num_epochs):
    model.eval()

    with torch.no_grad():
        for image_num, (val_images, val_labels) in tqdm(enumerate(val_loader)):

            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            
            # model prediction
            val_pred = model(val_images)

            # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
            # considering predictions with highest scores for each pixel among 19 classes        
            prediction = val_pred.data.max(1)[1].cpu().numpy()
            ground_truth = labels_val.data.cpu().numpy()

            # replace 100 to change number of images to print. 
            # 500 % 100 = 5. So, we will get 5 predictions and ground truths
            if image_num % 100 == 0:
                
                # Model Prediction
                decoded_pred = val_data.decode_segmap(prediction[0])
                plt.imshow(decoded_pred)
                plt.show()
                plt.clf()
                
                # Ground Truth
                decode_gt = val_data.decode_segmap(ground_truth[0])
                plt.imshow(decode_gt)
                plt.show()

if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U_Net_model(3, 19).to(device)
