import torch
import spacy
import torch.nn as nn
from torchtext.data import Field

class Transformer(nn.Module):
    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoder, num_decoder, feedforward, dropout, max_len, device,):
        super(Transformer, self).__init__()

        #embded input sentence and positional encoding
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        
        #embded output sentence and positional encoding
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device

        #initialize torch's Transformer
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder, num_decoder, feedforward, dropout,)
        #final linear transformation for outputs
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        #dropout for regularization
        self.dropout = nn.Dropout(dropout)
        #padding for input index
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        #masking to avoid model looking ahead
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_length, N = src.shape
        trg_length, N = trg.shape

        #get positional encoding
        src_positions = (torch.arange(0, src_length).unsqueeze(1).expand(src_length, N).to(self.device))
        trg_positions = (torch.arange(0, trg_length).unsqueeze(1).expand(trg_length, N).to(self.device))

        #get embeddings for input  and output with drop
        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))

        #apply mask
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_length).to(self.device)

        #input data to model
        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask,)
        
        #obtain final predictions from last layer previously defined
        out = self.fc_out(out)
        return out

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def translate_sentence(sentence, german, english, device, max_length, trans):
    # Load tokenizer
    if trans == "en_de":
      spacy_eng = spacy.load("en")
    elif trans == "de_en":
      spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        if trans == "en_de":
          tokens = [token.text.lower() for token in spacy_eng(sentence)]
        elif trans == "de_en":
          tokens = [token.text.lower() for token in spacy_ger(sentence)]
        # tokens = [token.text.lower() for token in spacy_fr(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    if trans == "en_de":
      tokens.insert(0, english.init_token)
      tokens.append(english.eos_token)
      # Go through each german token and convert to an index
      text_to_indices = [english.vocab.stoi[token] for token in tokens]
    elif trans == "de_en":
      tokens.insert(0, german.init_token)
      tokens.append(german.eos_token)
      # Go through each german token and convert to an index
      text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    if trans == "en_de":
      outputs = [german.vocab.stoi["<sos>"]]
    elif trans == "de_en":
      outputs = [english.vocab.stoi["<sos>"]]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if trans == "en_de":
          if best_guess == german.vocab.stoi["<eos>"]:
            break
        elif trans == "de_en":
          if best_guess == english.vocab.stoi["<eos>"]:
            break

    if trans == "en_de":
      translated_sentence = [german.vocab.itos[idx] for idx in outputs]
    elif trans == "de_en":
      translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token        
    return translated_sentence[1:]

def prediction(sentence):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  #Translate from english to german
  # trans = "en_de"

  #Translate from german to english
  trans = "en_de"  
  
  english = Field(tokenize=tokenize_eng, init_token="<sos>", eos_token="<eos>", lower=True, )
  german = Field(tokenize=tokenize_ger, init_token="<sos>", eos_token="<eos>", lower=True)
  english.vocab = torch.load('model/vocab_eng.pth')
  german.vocab = torch.load('model/vocab_germ.pth')

  #Get size of vocabulary
  if trans == "en_de":
      src_vocab_size = len(english.vocab)
      trg_vocab_size= len(german.vocab)
  elif trans == "de_en":
      src_vocab_size = len(german.vocab)
      trg_vocab_size = len(english.vocab)
  else:
      print("Please go to previous cell and choose between en_de or de_en")  

  #Input size into the model
  embedding_size = 512

  #Number of multihead attentions
  num_heads = 8

  #Number of encoders and decoders
  num_encoder = 3
  num_decoder = 3
  dropout = 0.10

  #Max lenght of each sentence
  max_len = 100
  feedforward = 4

  if trans == "en_de":
      src_pad_idx = german.vocab.stoi["<pad>"]
  elif trans == "de_en":
      src_pad_idx = english.vocab.stoi["<pad>"]

  #Initialize model with model hyperparameters
  model = Transformer(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, 
                          num_encoder, num_decoder, feedforward, dropout, max_len, device).to(device)

  # Specify a path
  PATH = "model/en_de_model.pt"

  # Load
  model = torch.load(PATH)

  model.eval()
  # sentence = "a man in an orange hat starring at something."
  translated_sentence = translate_sentence( model, sentence, german, english, device, max_length=50, trans=trans)

  return translated_sentence