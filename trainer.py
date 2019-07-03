import torch
from test import Tester
from utils import MetricTracker


class Trainer:
    def __init__(self, config, model, optimizer, criterion, dataloader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

        self.losses = MetricTracker()
        self.accs = MetricTracker()

        self.tester = Tester(self.config, self.model)

    def train(self):
        for epoch in range(self.config.num_epochs):
            result = self._train_epoch(epoch)
            print('Epoch: [{0}]\t Avg Loss {loss:.4f}\t Avg Accuracy {acc:.3f}'.format(epoch, loss=result['loss'], acc=result['acc']))
            # NOTE MODIFICATION (TEST)
            self.tester.eval()

            if self.tester.best_acc == self.tester.accs.avg:
                print('Saving Model...')
                torch.save({
                    'epoch': epoch,
                    'model': self.model,
                    'optimizer': self.optimizer,
                }, 'best_model/model.pth.tar')

    def _train_epoch(self, epoch_idx):
        self.model.train()

        self.losses.reset()
        self.accs.reset()

        for batch_idx, (docs, labels, doc_lengths, sent_lengths) in enumerate(self.dataloader):
            batch_size = labels.size(0)

            docs = docs.to(self.device)  # (batch_size, padded_doc_length, padded_sent_length)
            labels = labels.to(self.device)  # (batch_size)
            sent_lengths = sent_lengths.to(self.device)  # (batch_size, padded_doc_length)
            doc_lengths = doc_lengths.to(self.device)  # (batch_size)

            # (n_docs, n_classes), (n_docs, max_doc_len_in_batch, max_sent_len_in_batch), (n_docs, max_doc_len_in_batch)
            scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)

            # NOTE MODIFICATION (BUG)
            self.optimizer.zero_grad()

            loss = self.criterion(scores, labels)
            loss.backward()

            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # NOTE MODIFICATION (BUG): clip grad norm should come before optimizer.step()
            self.optimizer.step()

            # Compute accuracy
            predictions = scores.max(dim=1)[1]
            correct_predictions = torch.eq(predictions, labels).sum().item()
            acc = correct_predictions

            self.losses.update(loss.item(), batch_size)
            self.accs.update(acc, batch_size)

            print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f}(avg: {loss.avg:.4f})\t Acc {acc.val:.3f} (avg: {acc.avg:.3f})'.format(
                    epoch_idx, batch_idx, len(self.dataloader), loss=self.losses, acc=self.accs))

        log = {'loss': self.losses.avg, 'acc': self.accs.avg}
        return log


