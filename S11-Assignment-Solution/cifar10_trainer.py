"""."""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time


def update_lr(optimizer, lr):
    """Update learning rate."""
    for g in optimizer.param_groups:
        g['lr'] = lr


def update_mom(optimizer, mom):
    """Update momentum."""
    for g in optimizer.param_groups:
        g['momentum'] = mom


def accuracy(output, target, is_test=False):
    global total
    global correct
    batch_size = target.size(0)
    total += batch_size
    _, pred = torch.max(output, 1)
    if is_test:
        preds.extend(pred)
    correct += (pred == target).sum()
    return 100 * correct / total


class AvgStats(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses =[]
        self.precs =[]
        self.its = []
        
    def append(self, loss, prec, it):
        self.losses.append(loss)
        self.precs.append(prec)
        self.its.append(it)


total = 0
correct = 0

train_loss = 0
test_loss = 0
best_acc = 0
trn_losses = []
trn_accs = []
val_losses = []
val_accs = []
preds =[]
train_stats = AvgStats()
test_stats = AvgStats()


def save_checkpoint(model, is_best, filename="model.pth"):
    """Save checkpoint if a new best is achieved."""
    if is_best:
        torch.save(model.state_dict(), filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


def train(epoch=0, use_cycle=False, onecycle=None, model=None, device="cuda",
          optimizer=None, criterion=None, train_loader=None):
    """Train the model."""
    model.train()
    global best_acc
    global trn_accs
    global trn_losses
    # is_improving = True
    # counter = 0
    running_loss = 0.
    avg_beta = 0.98

    for i, (input, target) in enumerate(train_loader):
        bt_start = time.time()
        input, target = input.to(device), target.to(device)
        var_ip, var_tg = Variable(input), Variable(target)

        if use_cycle:
            lr, mom = onecycle.calc()
            update_lr(optimizer, lr)
            update_mom(optimizer, mom)

        output = model(var_ip)
        loss = criterion(output, var_tg)
        running_loss = (
            avg_beta * running_loss
            + (1 - avg_beta) * loss.item()
        )
        smoothed_loss = running_loss / (1 - avg_beta**(i + 1))

        trn_losses.append(smoothed_loss)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        trn_accs.append(prec)

        train_stats.append(smoothed_loss, prec, time.time()-bt_start)
        if prec > best_acc:
            best_acc = prec
            save_checkpoint(model, True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, device, criterion, test_loader):
    """Test the model.."""
    model.eval()
    global val_accs
    global val_losses
    running_loss = 0.
    avg_beta = 0.98
    for i, (input, target) in enumerate(test_loader):
        bt_start = time.time()
        input, target = input.to(device), target.to(device)
        var_ip = Variable(input, volatile=True)
        var_tg = Variable(target, volatile=True)
        output = model(var_ip)
        loss = criterion(output, var_tg)

        running_loss = (
            avg_beta * running_loss
            + (1 - avg_beta) * loss.item()
        )
        smoothed_loss = running_loss / (1 - avg_beta ** (i + 1))

        # measure accuracy and record loss
        prec = accuracy(output.data, target, is_test=True)
        test_stats.append(loss.item(), prec, time.time()-bt_start)

        val_losses.append(smoothed_loss)
        val_accs.append(prec)


def fit(num_epochs=24, use_onecycle=False, onecycle=None, model=None,
        device="cuda", optimizer=None, criterion=None,
        train_loader=None, test_loader=None):
    """."""
    print("Epoch\tTrn_loss\tVal_loss\tTrn_acc\t\tVal_acc")
    for j in range(num_epochs):
        train(
            epoch=j,
            use_cycle=use_onecycle,
            onecycle=onecycle,
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader
        )
        test(model, device, criterion, test_loader)
        print(
            "{}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}"
            .format(
                j+1,
                trn_losses[-1],
                val_losses[-1],
                trn_accs[-1],
                val_accs[-1]
            )
        )

    return train_stats, test_stats
