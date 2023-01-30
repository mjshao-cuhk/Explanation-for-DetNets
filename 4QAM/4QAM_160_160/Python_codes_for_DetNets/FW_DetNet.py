
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as io
import numpy as np
import funcs


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dev = ("cpu")
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(5)
np.random.seed(0)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
dev = ("cpu")

pi = 3.14159265359
M = 160
N = 160
QAM_size = torch.tensor(4.).to(dev)
u = (torch.sqrt(QAM_size)-1).to(dev) 
Es=2./3.*(QAM_size-1)
constellation = ( torch.tensor(range(1, int(u)+2))*2-u-2 ).to(dev)

snrdb_low = 8.
snrdb_high = 20.

class FW_unfold(nn.Module):
     def __init__(self, cons, M, N, L):
          super().__init__()
          self.cons = cons
          self.s = cons.size()[0]
          self.N = N
          self.M = M
          self.L = L

          temp = []
          for ii in range(self.L):
               temp.append(0.0001)
          self.grad_ss = nn.Parameter(torch.tensor(temp))

          self.extra_ss = nn.Parameter(torch.ones(self.L) * 0.01)

          temp = []
          for ii in range(self.L):
               temp.append((2.-0.1)/self.L*ii+0.1)
          self.gamma = nn.Parameter(torch.tensor(temp))


     def forward(self, x_ini, y, H, xt):
          x = x_ini.clone()
          y_x = x_ini.clone()
          HTH = torch.matmul(H.transpose(1, 2), H)
          HTy = torch.matmul(H.transpose(1, 2), y)
          loss = torch.zeros(1, 1).to(dev)
          for ii in range(self.L):
              grad_f = 2*(torch.matmul(HTH, y_x)-HTy)
              x_buff = y_x - self.grad_ss[ii]*grad_f
              temp = 2.*self.gamma[ii]*x_buff
              ply_x = 2*torch.sigmoid(temp)-1  

              y_x =  x + self.extra_ss[ii] * (ply_x - x)
              x = y_x.clone()
          loss =  torch.sum(torch.matmul((x - xt).transpose(1, 2), (x - xt)), 0)
          return x, loss

num_iter = 100000
batch_size = 200
adam_lr = 0.001
layers = 20
model = FW_unfold(constellation, M, N, layers).to(dev)
for param in model.parameters():
    print(param.data)
opt = optim.Adam(model.parameters(), lr=adam_lr)


def adjust_learning_rate(optimizer, epoch, ini_lr):
     lr = ini_lr * (0.97 ** (epoch // 1000))
     for param_group in optimizer.param_groups:
          param_group['lr'] = lr


model.train()
for epoch in range(num_iter):  # num of  train iter
     X_c, train_X, Y_c, train_Y, H_c, train_H, noise_c, noise, train_sigma2 = funcs.generate_data(batch_size, int(N / 2),
                                                                                        int(M / 2), snrdb_low,
                                                                                        snrdb_high, u, Es)
     x_ini = torch.zeros(batch_size, N, 1).double()
     opt.zero_grad()
     x_est, loss = model(x_ini, train_Y, train_H, train_X)
     loss = loss/batch_size
     loss.backward()


     opt.step()
     adjust_learning_rate(opt, epoch, adam_lr)

     if epoch % 20 == 0:
          s_FW_unfold = funcs.de_QAM(x_est, constellation, 1, N).numpy().reshape(N, -1)
          s_FW_unfold = s_FW_unfold.reshape(batch_size, N, 1)
          sc_FW_unfold = s_FW_unfold[:, 0:int(N / 2), :] + 1j * s_FW_unfold[:, int(N / 2):int(N), :]
          BR_count = funcs.BR(sc_FW_unfold, X_c)
          BRE_err = 1 - BR_count / (N / 2 * batch_size)
          print("loss after ", epoch, "th batch", loss, "BER", BRE_err, "learning rate:",
                opt.state_dict()['param_groups'][0]['lr'])

     # if epoch%200 ==0:
     #     for param in model.parameters():
     #         print(param.grad)

     # if epoch % 1000 == 0:
     #      # save parameters
     #      filename1 = 'your path'
     #      filename2 = 'FW_parameter' + str(int(QAM_size)) + "Q" + str(M) + "_" + str(N) + "_" + "SNR" + str(
     #           snrdb_low) + "_" + str(snrdb_high) + "_" + str(epoch) + "uni_dB.pkl"
     #      filename = filename1 + filename2
     #      torch.save(model.state_dict(), filename)

