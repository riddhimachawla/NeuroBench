import os
import torch


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=50,
    ):

        self.model = model.to(device)

        self.train_loader = train_loader

        self.test_loader = test_loader

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = device

        self.epochs = epochs

        self.best_accuracy = 0.0

        self.train_loss_history = []

        self.train_accuracy_history = []

        self.test_accuracy_history = []

        os.makedirs("models", exist_ok=True)

    # -----------------------------------------------------

    def train(self):

        print("=" * 60)
        print("Training EEGNet")
        print("=" * 60)

        for epoch in range(self.epochs):

            train_loss, train_acc = self.train_one_epoch()

            test_acc = self.evaluate()

            self.train_loss_history.append(train_loss)

            self.train_accuracy_history.append(train_acc)

            self.test_accuracy_history.append(test_acc)

            print(
                f"Epoch {epoch+1:03d}/{self.epochs}"
                f" | Loss {train_loss:.4f}"
                f" | Train {train_acc:.2f}%"
                f" | Test {test_acc:.2f}%"
            )

            if test_acc > self.best_accuracy:

                self.best_accuracy = test_acc

                torch.save(

                    self.model.state_dict(),

                    "models/eegnet_best.pt"

                )

        print()

        print("=" * 60)

        print("Training Finished")

        print("=" * 60)

        print(f"Best Accuracy : {self.best_accuracy:.2f}%")

    # -----------------------------------------------------

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0

        correct = 0

        total = 0

        for X, y in self.train_loader:

            X = X.to(self.device)

            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(X)

            loss = self.criterion(

                outputs,

                y

            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += y.size(0)

            correct += predicted.eq(y).sum().item()

        epoch_loss = running_loss / len(self.train_loader)

        accuracy = 100 * correct / total

        return epoch_loss, accuracy

    # -----------------------------------------------------

    @torch.no_grad()

    def evaluate(self):

        self.model.eval()

        correct = 0

        total = 0

        for X, y in self.test_loader:

            X = X.to(self.device)

            y = y.to(self.device)

            outputs = self.model(X)

            _, predicted = outputs.max(1)

            total += y.size(0)

            correct += predicted.eq(y).sum().item()

        accuracy = 100 * correct / total

        return accuracy

    # -----------------------------------------------------

    def get_history(self):

        return {

            "loss": self.train_loss_history,

            "train_accuracy": self.train_accuracy_history,

            "test_accuracy": self.test_accuracy_history

        }