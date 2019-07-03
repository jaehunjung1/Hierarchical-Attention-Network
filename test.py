from dataset import News20Dataset, collate_fn
from utils import *
import os, sys
import webbrowser


class Tester:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = next(self.model.parameters()).device

        self.dataset = News20Dataset(config.cache_data_dir, config.vocab_path, is_train=False)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False,
                                                      collate_fn=collate_fn)

        self.accs = MetricTracker()
        self.best_acc = 0

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            self.accs.reset()

            for (docs, labels, doc_lengths, sent_lengths) in self.dataloader:
                batch_size = labels.size(0)

                docs = docs.to(self.device)
                labels = labels.to(self.device)
                doc_lengths = doc_lengths.to(self.device)
                sent_lengths = sent_lengths.to(self.device)

                scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)

                predictions = scores.max(dim=1)[1]
                correct_predictions = torch.eq(predictions, labels).sum().item()
                acc = correct_predictions

                self.accs.update(acc, batch_size)
            self.best_acc = max(self.best_acc, self.accs.avg)

            print('Test Average Accuracy: {acc.avg:.4f}'.format(acc=self.accs))


if __name__ == "__main__":
    if not os.path.exists("best_model/model.pth.tar"):
        print("Visualization requires pretrained model to be saved under ./best_model.\n")
        print("Please run 'python train.py <args>'")
        sys.exit()

    checkpoint = torch.load("best_model/model.pth.tar")
    model = checkpoint['model']
    model.eval()

    dataset = News20Dataset("data/news20/", "data/glove/glove.6B.100d.txt", is_train=False)
    doc = "First of all, realize that Tesla invented AC power generators, motors,\
    transformers, conductors, etc. Technically, *ALL* transformers are Tesla\
    coils.  In general though when someone refers to a Tesla coil, they mean\
    an 'air core resonant transformers'."

    result = visualize(model, dataset, doc)

    with open('result.html', 'w') as f:
        f.write(result)

    webbrowser.open_new('file://'+os.getcwd()+'/result.html')
