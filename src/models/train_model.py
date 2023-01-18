import torch
from torch import optim, nn
from src.models.predict_model import validation
import hydra

@hydra.main(config_path="config", config_name='config.yaml')
def train(model, trainloader, testloader, config, print_every=40):
    hparams = config.experiment
    steps = 0
    running_loss = 0
    loss_total = []    

    optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"]) 
    criterion = nn.CrossEntropyLoss()

    for e in range(hparams["epochs"]):
        # Model in training mode, dropout is on
        model.train()

        for images, labels in trainloader:
            steps += 1
            images = images.float()
            labels = labels.long()
            
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                
            # print(images.shape)
            # print(labels)
            optimizer.zero_grad()
            
            output = model.forward(images)
            # print(output.shape)
            loss = criterion(output, labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loss_total.append(loss.item())
            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, hparams["epochs"]),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                                    
                running_loss = 0
                
                # Make suredropout and grads are on for training
                model.train()
    
    return model