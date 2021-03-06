"""."""
import torch
import torch.nn.functional as F
from tqdm import tqdm

class ModelTrainer:
    """."""
    def __init__(self):
        """."""
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train(self, model, device, train_loader, optimizer, epoch, criterion, l1_penalty=0):
        """Train and get accuracy.."""
        model.train()
        pbar = train_loader#tqdm(train_loader)
        correct = 0
        processed = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            # finding batch size 
            m = data.shape[0]
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, target)

            if l1_penalty > 0:
                l1 =0
                for p in model.parameters():
                    l1+=p.abs().sum()
                loss += l1_penalty/(2*m)*l1

            train_loss += loss
            # self.train_losses.append(loss)
            loss.backward()
            optimizer.step()
            
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            # pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        
        train_loss /= len(train_loader.dataset)
        self.train_losses.append(train_loss if isinstance(train_loss,float) else train_loss.item())
        self.train_acc.append(100.0*correct/processed)
        print(
                "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    train_loss, 
                    correct, 
                    len(train_loader.dataset),
                    100. * correct / processed
                )
            )
   
        return train_loss

    def test(self, model, device, test_loader, criterion, l1_penalty=0):
        """Test and get test accuracy."""
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                m = data.shape[0]
                output = model(data)
                loss = criterion(output, target).item()  # sum up batch loss
                if l1_penalty > 0:
                    l1 =0
                    for p in model.parameters():
                        l1+=p.abs().sum()

                    loss += l1_penalty/(2*m)*l1

                test_loss += loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            self.test_losses.append(test_loss if isinstance(test_loss,float) else test_loss.item())
            print(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    test_loss, 
                    correct, 
                    len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)
                )
            )
            self.test_acc.append(100. * correct / len(test_loader.dataset))
            return test_loss
