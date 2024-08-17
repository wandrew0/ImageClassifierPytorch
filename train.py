import torch
from torch import nn
import QuickDraw as Data
from vgg5 import Vgg5
from kaggleguy import KaggleGuy2
import time

# torch.manual_seed(0)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def train_loop(dataloader, model, loss_fn, optimizer, augment=lambda x: x):
    num_batches = len(dataloader)
    print_interval = -(num_batches // -10)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        X = augment(X)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_interval == 0:
            loss = loss.item()
            print(
                f"| loss: {loss:>7f} [{batch + 1}/{num_batches}] |",
                end="\t",
                flush=True,
            )


def test_loop(dataloader, model, loss_fn, augment=lambda x: x):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            X = augment(X)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"\nTest Error: \nAccuracy: {(100*correct):>0.2f}%, Loss: {test_loss:>8f}")

    return correct * 100


batch_size = 64

train_dataloader = Data.train_dataloader(batch_size)
test_dataloader = Data.test_dataloader(batch_size)
augment = Data.augment()

model = KaggleGuy2(Data.input_size, Data.num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params:,}")
loss_fn = nn.CrossEntropyLoss()
learning_rate = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# state = torch.load("./checkpoints/temp-KaggleGuy-EMNIST_balanced")
# model.load_state_dict(state["model"])
# optimizer.load_state_dict(state["optimizer"])

max_epochs = 100
quit = -10
delta = 0.01

no_improve_epochs = 0
best = 0
# best = state["best"]
for epoch in range(max_epochs):
    start = time.time()

    print(f"\nEpoch {epoch + 1}\n---------------------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, augment)
    test_acc = test_loop(test_dataloader, model, loss_fn)
    end = time.time()

    duration = end - start
    print(
        f"Completed in {duration:.2f} seconds. \t~{((len(test_dataloader.dataset) + len(train_dataloader.dataset)) / duration):.2f} img/s"
    )
    if test_acc < best + delta:
        no_improve_epochs += 1
    else:
        no_improve_epochs = 0
    if no_improve_epochs == quit:
        break
    if test_acc > best:
        best = test_acc
        filename = model.name + "-" + Data.name + str(round(best * 100))
        print(
            f'New best model found at epoch {epoch}, saving model with name "{filename}".'
        )
        cp = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best": best,
        }
        torch.save(cp, "./checkpoints/" + filename)

if no_improve_epochs == quit:
    print(
        f"Reached {no_improve_epochs} epochs without improvement over {best:>.2f}, exiting."
    )
else:
    print(f"{max_epochs} epochs reached, done.")

cp = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "best": best,
}

filename = "temp-" + model.name + "-" + Data.name
torch.save(cp, "./checkpoints/" + filename)
print(f'Saving final model with name "{filename}".')
