import torch
from src.models.predict_model import validation


def train(
    model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40
):
    steps = 0
    running_loss = 0
    loss_total = []

    for e in range(epochs):
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

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                running_loss = 0

                # Make suredropout and grads are on for training
                model.train()


def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0

    for images, labels in testloader:

        # images = images.resize_(images.size()[0], 784)
        images = images.float()
        labels = labels.long()

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        pred = ps.max(1)[1]
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == pred
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
