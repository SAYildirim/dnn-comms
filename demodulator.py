import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------- Parameters ----------
n_symbols = 500000           # total symbols to generate
train_frac = 0.8            # fraction for training
snr_db_train = 10.0          # training SNR in dB (Eb/N0-like)
snr_db_test = 3.0           # testing SNR in dB
batch_size = 64
n_epochs = 100
lr = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Helper functions ----------
def psk_constellation(M):
    # M-PSK constellation generation
    amplitudes = np.ones(M)  # unit circle
    angles     = np.arange(M) * 2 * np.pi / M
    return amplitudes * np.exp(1j * angles)

def qam_constellation(M):
    # M-QAM constellation generation
    pass

def bpsk_map(bits):
    # Map bits {0,1} -> symbols {-1,+1}
    return 2 * bits - 1

def awgn_iq(symbols_iq, snr_db):
    # symbols_iq: shape (N, 2) real (I,Q)
    # compute noise std from SNR (assuming symbol power = mean(|s|^2))
    N0 = 1 # assume unit power for noise spectral density (per dimension)
    pn = 2*N0
    ps = np.mean(np.sum(symbols_iq ** 2, axis=1))
    snr_linear = 10 ** (snr_db / 10.0)
    ps *= N0/2 * snr_linear;
    sigma = np.sqrt(pn / 2)  # per-dimension (I and Q) variance
    noise = sigma * np.random.randn(*symbols_iq.shape)
    symbols_iq = np.sqrt(ps) * symbols_iq
    return symbols_iq + noise

class PSKModem(Modem):
    # Derived class: PSKModem
    def __init__(self, M):
        #Generate reference constellation
        m = np.arange(0,M) #all information symbols m={0,1,...,M-1}
        I = 1/np.sqrt(2)*np.cos(m/M*2*np.pi)
        Q = 1/np.sqrt(2)*np.sin(m/M*2*np.pi)
        constellation = I + 1j*Q #reference constellation
        Modem.__init__(self, M, constellation, name='PSK') #set the modem attributes

class QAMModem(Modem):
    # Derived class: QAMModem
    def __init__(self,M):
        if (M==1) or (np.mod(np.log2(M),2)!=0): # M not a even power of 2
            raise ValueError('Only square MQAM supported. M must be even power of 2')
        
        n = np.arange(0,M) # Sequential address from 0 to M-1 (1xM dimension)
        a = np.asarray([xˆ(x>>1) for x in n]) #convert linear addresses to Gray code
        D = np.sqrt(M).astype(int) #Dimension of K-Map - N x N matrix
        a = np.reshape(a,(D,D)) # NxN gray coded matrix
        oddRows=np.arange(start = 1, stop = D ,step=2) # identify alternate rows
        a[oddRows,:] = np.fliplr(a[oddRows,:]) #Flip rows - KMap representation
        nGray=np.reshape(a,(M)) # reshape to 1xM - Gray code walk on KMap
        #Construction of ideal M-QAM constellation from sqrt(M)-PAM
        (x,y)=np.divmod(nGray,D) #element-wise quotient and remainder
        Ax=2*x+1-D # PAM Amplitudes 2d+1-D - real axis
        Ay=2*y+1-D # PAM Amplitudes 2d+1-D - imag axis
        constellation = Ax+1j*Ay
        Modem.__init__(self, M, constellation, name='QAM') #set the modem attributes





# ---------- Generate dataset ----------
# Random bits
bits = np.random.randint(0, 2, size=(n_symbols,)).astype(np.float32)
# Map to BPSK symbols on I axis; Q = 0
symbols = bpsk_map(bits)  # shape (n_symbols,)
symbols_iq = np.stack([symbols, np.zeros_like(symbols)], axis=1)  # (N,2)

# Add AWGN for training and testing sets separately (optionally can vary SNR)
# We'll create a single noisy dataset at chosen SNR for training and testing split.
noisy_symbols = awgn_iq(symbols_iq, snr_db_train)

# Split train/test
n_train = int(n_symbols * train_frac)
train_x = noisy_symbols[:n_train]
train_y = bits[:n_train]
test_x = awgn_iq(symbols_iq[n_train:], snr_db_test)  # re-noise test partition (fresh noise)
test_y = bits[n_train:]

# Convert to PyTorch tensors
train_x_t = torch.from_numpy(train_x).float()
train_y_t = torch.from_numpy(train_y).float().long()  # shape (N,1)
test_x_t = torch.from_numpy(test_x).float()
test_y_t = torch.from_numpy(test_y).float().long()

train_ds = TensorDataset(train_x_t, train_y_t)
test_ds = TensorDataset(test_x_t, test_y_t)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# ---------- Single-layer model ----------
class SingleLayerDemod(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 2 -> two logits -> sigmoid gives probability of bits
        self.linear = nn.Linear(2, 2, bias=True)

    def forward(self, x):
        # x shape: (batch, 2)
        out = self.linear(x)  # (batch,2)
        prob = torch.softmax(out, dim=1)
        return prob

model = SingleLayerDemod().to(device)

# Initialize weights sensibly (small)
nn.init.normal_(model.linear.weight, mean=0.0, std=0.1)
nn.init.constant_(model.linear.bias, 0.0)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# ---------- Training ----------
for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        probs = model(xb)
        loss = criterion(probs, nn.functional.one_hot(yb).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / n_train

    # Evaluate on test set each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            probs = model(xb)
            preds = (probs >= 0.5).float()
            correct += (preds == nn.functional.one_hot(yb).float()).sum().item()/2
            total += yb.numel()
    acc = correct / total
    W_ = model.linear.weight.cpu().detach().view(2,-1).numpy()
    b = model.linear.bias.cpu().detach().numpy()
    print(f"Epoch {epoch}/{n_epochs}  Loss {epoch_loss:.4f}  Test Acc {acc*100:.2f}%")
    if epoch == n_epochs or epoch == 1:
        print(f"Weights: w11 = {W_[0,0]:.4f}  w12 = {W_[0,1]:.4f}  w21 = {W_[1,0]:.4f}  w22 = {W_[1,1]:.4f} \
        Bias: b1 = {b[0]:.4f} b2 = {b[1]:.4f}")

# ---------- Final evaluation ----------
model.eval()
with torch.no_grad():
    xb = test_x_t.to(device)
    yb = test_y_t.to(device)
    probs = model(xb)
    preds = (probs >= 0.5).float()
    acc_final = (preds[:,1] == yb).float().mean().item()
print(f"Final test accuracy: {acc_final*100:.2f}%")

# Show some example predictions (first 10)
examples = 10
with torch.no_grad():
    ex_x = xb[:examples].cpu().numpy()
    ex_y = yb[:examples].cpu().numpy().flatten()
    ex_p = probs[:examples].cpu().numpy().flatten()
    ex_pred = preds[:examples].cpu().numpy().flatten()

print("\nExample (I, Q) | true_bit -> prob -> pred_bit")
for i in range(examples):
    i_val, q_val = ex_x[i]
    print(f"({i_val: .3f}, {q_val: .3f}) | {int(ex_y[i])} -> {ex_p[i]:.3f} -> {int(ex_pred[i])}")