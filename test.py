import EMNIST as Data

train_dataloader = Data.train_dataloader(32)

print(next(iter(train_dataloader))[0].shape)

# for b, (X, y) in enumerate(train_dataloader):
#     X = X + 1
