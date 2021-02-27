"""Run LR range test."""
from CLR import CLR
from torch.autograd import Variable


def update_lr(optimizer, lr):
    """Update learning rate."""
    for g in optimizer.param_groups:
        g['lr'] = lr


def update_mom(optimizer, mom):
    """Update momentum."""
    for g in optimizer.param_groups:
        g['momentum'] = mom


def run_learning_rate_range_test(model, criterion, optimizer,
                                 train_loader, device):
    """Run LR range test."""
    running_loss = 0.
    avg_beta = 0.98
    model.train()
    clr = CLR(optimizer, len(train_loader))
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        var_ip, var_tg = Variable(input), Variable(target)
        output = model(var_ip)
        loss = criterion(output, var_tg)
        running_loss = avg_beta * running_loss + (1 - avg_beta) * loss.item()
        smoothed_loss = running_loss / (1 - avg_beta**(i + 1))
        lr = clr.calc_lr(smoothed_loss)
        if lr == -1:
            break
        update_lr(optimizer, lr)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    clr.plot()
