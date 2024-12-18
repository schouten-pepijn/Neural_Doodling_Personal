import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys


#%% VARIABLES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 512
seq_len = 80
num_layers = 3
epochs = 10
num_epochs = 20

#%% DATA
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus augue nunc, ullamcorper eu vestibulum eget, elementum ac sapien. Morbi eget porttitor tellus. Sed ac laoreet tellus. Suspendisse tincidunt bibendum erat non scelerisque. Maecenas vel sagittis nulla, ac sollicitudin tellus. Proin aliquam leo id libero scelerisque sollicitudin. Proin et arcu consequat, viverra lectus in, consectetur urna. Sed tempus accumsan est, quis tristique lectus mollis eu. Cras id porta libero. Integer sem ante, sollicitudin a ex quis, gravida imperdiet turpis. Proin euismod sapien ac ornare elementum. Aenean eu imperdiet velit, at tempor erat. Ut et dignissim dolor. Mauris nec faucibus ipsum, eget rhoncus metus. Morbi iaculis sagittis metus, vel lobortis urna porttitor nec. Donec blandit lacus quis posuere porttitor. In tempor tincidunt elementum. Sed auctor pellentesque ex id aliquam. Pellentesque consectetur nulla sollicitudin lorem efficitur vestibulum non sed erat. Suspendisse at dignissim orci, eu ultrices nulla. Vivamus quis scelerisque felis. Aliquam ut ante leo. Mauris at vulputate magna. Nullam aliquet laoreet mauris, in ultricies mi commodo nec. Sed ultricies ipsum vel ex euismod, id malesuada magna tempus. Suspendisse ornare faucibus nisl, vitae viverra libero aliquam ac. Aenean tincidunt vulputate risus at dapibus. Sed porta diam at massa bibendum posuere. Curabitur ornare diam vitae pharetra dictum. Suspendisse potenti. Suspendisse dapibus elementum massa rhoncus mattis. Curabitur finibus nulla eu lectus fermentum, eget rutrum nisl finibus. Quisque condimentum tortor ut sem malesuada, sit amet imperdiet eros volutpat. Vivamus commodo ac libero eget convallis. Pellentesque quis interdum nulla. Nunc porta blandit aliquam. Phasellus nibh velit, cursus sit amet leo ac, porta interdum lacus. Duis vitae accumsan enim. Maecenas ac luctus sapien. Donec sagittis neque vitae est porta, non porttitor dolor sollicitudin. Donec commodo leo et efficitur ornare. Proin sollicitudin libero in tortor aliquet euismod. Etiam consequat erat vitae pharetra fringilla. Cras non eleifend libero, nec scelerisque nibh. Nunc tristique, mi at pretium placerat, leo dolor consequat libero, id rhoncus tortor nisi a tortor. Nulla tempor erat pharetra quam ullamcorper, sit amet sollicitudin tellus tincidunt. Duis tincidunt tellus vitae nibh cursus maximus. Nam porta finibus fringilla. Nunc sit amet risus ac lectus sagittis molestie. Sed et condimentum elit, luctus iaculis ipsum. Quisque augue risus, scelerisque eget velit id, maximus viverra magna. Nullam sit amet dolor nibh. Aliquam pulvinar sem massa, ut dictum quam consequat vehicula. Mauris semper luctus nulla nec mollis. Pellentesque ex nisi, fermentum quis lorem sed, venenatis imperdiet lectus. Maecenas auctor augue nec pellentesque dapibus. Aenean ornare dui vitae erat egestas vulputate. Duis convallis pharetra varius. Mauris vitae suscipit felis, ut sodales libero. Sed ut felis quis magna gravida malesuada vel ac neque. Aliquam fringilla justo purus, vel hendrerit lorem dapibus at. Praesent eleifend sapien ex, sed rutrum massa imperdiet id. Aliquam at dictum justo, et viverra justo. "
text = text.lower()

unique_chars = sorted(set(text))

num_2_let = dict(enumerate(unique_chars))
let_2_num = {l:i for i,l in num_2_let.items()}

data = torch.tensor([let_2_num[ch] for ch in text], dtype=torch.int64).unsqueeze(1)
data = data.to(device)

plt.figure(dpi=200)
plt.plot(data.cpu().numpy(), 'k.')
plt.xlabel("Character index")
plt.ylabel("Character label")
plt.show()


#%% MODEL
class LSTMmodel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=input_size,
                                      embedding_dim=input_size)
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h):
        
        embedding = self.embedding(x)
        
        y, h = self.lstm(embedding, h)
        
        y = self.out(y)
        
        return y, (h[0].detach(), h[1].detach())
        

#%% METAPARAMS
lstm_net = LSTMmodel(len(unique_chars), len(unique_chars),
                     hidden_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_net.parameters(), lr=1e-3)


#%% TESTING
I = next(lstm_net.embedding.named_parameters())

plt.figure()
plt.imshow(I[1].cpu().detach())
plt.show()


#%% TRAINING
losses = np.zeros(num_epochs)

lstm_net.train()
for epoch in range(num_epochs):

    txt_loss = 0
    hidden_state = None

    # loop over segments
    for txt_loc in range(0,len(text)-seq_len):
        x = data[txt_loc:txt_loc+seq_len]
        y = data[txt_loc+1:txt_loc+seq_len+1]

        # forward pass
        output, hidden_state = lstm_net(x, hidden_state)

        loss = criterion(output.squeeze(), y.squeeze())
        txt_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses[epoch] = txt_loss / txt_loc
 
    if (epoch+1) % 2 == 0:
        msg = f"Finished epoch {epoch+1}/{num_epochs} | " \
            f"Train loss: {losses[epoch]:.3f}"
        sys.stdout.write("\r" + msg)

#%% EVALUATION
plt.figure(dpi=200)
plt.plot(losses, 'ks-', label='Training loss')
plt.legend()
plt.show()