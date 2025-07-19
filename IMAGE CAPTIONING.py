import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):  # ✅ Make sure both 'self' and 'embed_size' are present
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # Remove the last FC layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        """
        Forward pass through the network.
        :param features: image features from the encoder, shape (batch_size, embed_size)
        :param captions: ground truth captions, shape (batch_size, caption_len)
        """
        embeddings = self.embed(captions)  # (batch_size, caption_len, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # prepend image feature
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=20):
        """
        Generate captions for given image features using greedy search.
        :param features: image features from the encoder, shape (1, embed_size)
        :return: predicted word indices, shape (1, max_len)
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)           # (1, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))             # (1, vocab_size)
            _, predicted = outputs.max(1)                         # (1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                        # (1, embed_size)
            inputs = inputs.unsqueeze(1)                          # (1, 1, embed_size)

        sampled_ids = torch.stack(sampled_ids, 1)  # (1, max_len)
        return sampled_ids
# -----------------------------------------------
# Image preprocessing function
# -----------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------
# Example usage
# -----------------------------------------------
# You can replace this path with your own image
image_path = r"C:\Users\sumit\OneDrive\Desktop\images (1).jpeg"
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # add batch dimension

# Hyperparameters (for demonstration)
embed_size = 256
hidden_size = 512
vocab_size = 5000  # Example vocab size — in practice, set to your tokenizer vocab size

# Initialize models
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
# Extract features
features = encoder(image)

# Dummy caption tensor for training example (here random, just to show shape)
dummy_captions = torch.randint(0, vocab_size, (1, 10))

# Forward pass (training)
outputs = decoder(features, dummy_captions)
print("Decoder outputs shape:", outputs.shape)

# Sampling caption (inference)
sampled_ids = decoder.sample(features)
print("Sampled IDs:", sampled_ids)
sampled_ids = torch.tensor([[2,3,4,5,6]])
# -----------------------------------------------
# Example: Convert IDs to words (need your tokenizer)
# -----------------------------------------------
# Example word map (fake for demo)
word_map = {0: '<pad>', 1: '<start>', 2: 'a', 3: 'dog', 4: 'on', 5: 'grass', 6: '<end>'}
sentence = []
for idx in sampled_ids[0]:
    word = word_map.get(idx.item(), '<unk>')
    if word == '<end>':
        break
    sentence.append(word)

print("Generated caption:", ' '.join(sentence))