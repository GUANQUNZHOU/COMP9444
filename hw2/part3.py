import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
#ADDED LIB
import re
# import spacy

# nltk.download('stopwords')
# nltk.download('wordnet')
# spacy_en = spacy.load('en')

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

def tokenizer(text): # create a tokenizer function
    text = remove_tags(text)
    text = re.sub(r'\'s',' is',text)#is
    text = re.sub(r'\'re',' are',text)#are
    text = re.sub(r'\'ve',' have',text)#have
    text = re.sub(r'\'d',' would',text)#would
    text = re.sub(r'\'ll',' will',text)#will
    text = re.sub(r'n\'t',' not',text)#not
    text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub(r' +',' ',text)
    text = re.sub(r'am|is|are',"be",text)#am,is,are->be
#     a = [tok.text for tok in spacy_en.tokenizer(text)]
#     print('a is ',a)
#     print('split is ',text.split(' '))
    text = text.split(' ')
    return text#[tok.text for tok in spacy_en.tokenizer(text)]

# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.lstm = tnn.LSTM(input_size=50,hidden_size=190,num_layers=2,batch_first=True,bidirectional=True)
        # print("Initing W .......")
#         tnn.init.orthogonal_(self.lstm.all_weights[0][0])
#         tnn.init.orthogonal_(self.lstm.all_weights[0][1])
#         tnn.init.orthogonal_(self.lstm.all_weights[1][0])
#         tnn.init.orthogonal_(self.lstm.all_weights[1][1])
        self.dropout = tnn.Dropout(p = 0.5)
        self.ln1 = tnn.Linear(190,64)
        self.ln2 = tnn.Linear(64,1)
       #==========85.5% with 10 epochs===========
    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """
        x_packed = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True,enforce_sorted=True)
        packed_output, (ht, ct) = self.lstm(x_packed)
        # print('putput ',packed_output)
        output_padded = ht[-1]

        y = tnn.functional.relu(self.ln1(output_padded))
        rst = torch.squeeze(self.ln2(y))
        return rst

        
class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        x_space = [word for word in x if word != ' ']
        return x_space

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        print('a batch ',batch)
        print('a vocab ',vocab)
        return batch, vocab
    #remove stop words
#     nltk_stopwords = nltk.corpus.stopwords.words('english')
#     stopwords = nltk_stopwords[:10]
    text_field = data.Field(lower=True,include_lengths=True, tokenize=tokenizer,batch_first=True, preprocessing=pre, postprocessing=None)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)
#     print('after field: ',textField,labelField)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")
#     print('train dev: ',train,dev)

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)
#     print('after build vocab: ',)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)
    
    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(15):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)
            
        
            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

#     # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
