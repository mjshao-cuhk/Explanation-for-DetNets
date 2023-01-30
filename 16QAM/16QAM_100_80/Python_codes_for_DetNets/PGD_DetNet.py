
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
setup_seed(6)
np.random.seed(0)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


pi = 3.14159265359
M = 100
N = 80
QAM_size = torch.tensor(16.).to(dev)
u = (torch.sqrt(QAM_size)-1).to(dev)  
Es=2./3.*(QAM_size-1)
constellation = ( torch.tensor(range(1, int(u)+2))*2-u-2 ).to(dev)

snrdb_low = 20.
snrdb_high = 30.


class PG_unfold(nn.Module):
     def __init__(self, cons, M, N, L):
          super().__init__()
          self.cons = cons
          self.s = cons.size()[0]
          self.N = N
          self.M = M
          self.L = L

          self.central =  torch.tensor([-2, 0, 2])
          self.central_ex = self.central.repeat(N,1).reshape(1, N, 3)

          temp = []
          for ii in range(self.L):
               temp.append(0.001)
          self.grad_ss = nn.Parameter(torch.tensor(temp))  

          temp = []
          t = 1.
          for ii in range(self.L):
               t1 = (1+np.sqrt(1.+4.*(t*t))) / 2.
               temp.append( (t-1)/t1 )
               t = t1
          self.extra_ss = nn.Parameter(torch.tensor(temp))

          temp = []
          for ii in range(self.L):
               temp.append((5.-0.1)/self.L*ii+0.1)
          self.gamma1 = nn.Parameter(torch.tensor(temp))


     def forward(self, x_ini, y, H, xt):
          x = x_ini.clone()
          y_x = x_ini.clone()
          HTH = torch.matmul(H.transpose(1, 2), H)
          HTy = torch.matmul(H.transpose(1, 2), y)
          loss = torch.zeros(1, 1).to(dev)
          for ii in range(self.L):
              grad_f = 2*(torch.matmul(HTH, y_x)-HTy)
              x_buff = y_x - self.grad_ss[ii]*grad_f

              (val, val_indx) = torch.min( torch.abs(x_buff-self.central_ex), 2)
              val_indx = val_indx.reshape(batch_size, N, 1)
              center = self.central[val_indx]
              ply_x = 2*torch.sigmoid(self.gamma1[ii]*(x_buff-center))-1

              y_x = ply_x + self.extra_ss[ii] * (ply_x - x)
              x = ply_x.clone()

          loss =  torch.sum(torch.matmul((x - xt).transpose(1, 2), (x - xt)), 0)
          return x, loss


num_iter = 100000
batch_size = 1000
adam_lr = 0.0001
layers = 20  
model = PG_unfold(constellation, M, N, layers).to(dev)
for param in model.parameters():
    print(param.data)
opt = optim.Adam(model.parameters(), lr=adam_lr)


def adjust_learning_rate(optimizer, epoch, ini_lr):
     lr = ini_lr * (0.985 ** (epoch // 1000))
     for param_group in optimizer.param_groups:
          param_group['lr'] = lr

model.train()
for epoch in range(num_iter):  # num of  train iter
     X_c, train_X, Y_c, train_Y, H_c, train_H, noise_c, noise, train_sigma2 = funcs.generate_data(batch_size, int(N / 2),
                                                                                        int(M / 2), snrdb_low,
                                                                                        snrdb_high, u, Es)
     x_ini = torch.linalg.pinv(train_H) @ train_Y
     opt.zero_grad()
     x_est, loss = model(x_ini, train_Y, train_H, train_X)
     loss = loss/batch_size
     loss.backward()

     opt.step()
     adjust_learning_rate(opt, epoch, adam_lr)

     if epoch % 5 == 0:
          s_PG_unfold = funcs.de_QAM(x_est, constellation, 1, N).numpy().reshape(N, -1)
          s_PG_unfold = s_PG_unfold.reshape(batch_size, N, 1)
          sc_PG_unfold = s_PG_unfold[:, 0:int(N / 2), :] + 1j * s_PG_unfold[:, int(N / 2):int(N), :]
          BR_count = funcs.BR(sc_PG_unfold, X_c)
          BRE_err = 1 - BR_count / (N / 2 * batch_size)
          print("loss after ", epoch, "th batch", loss, "BER", BRE_err, "learning rate:",
                opt.state_dict()['param_groups'][0]['lr'])

     if epoch%200 ==0:
         for param in model.parameters():
             print(param.data)

     # if epoch % 1000 == 0:
     #      # save parameters
     #      filename1 = 'your path'
     #      filename2 = 'net_parameter' + str(int(QAM_size)) + "Q" + str(M) + "_" + str(N) + "_" + "SNR" + str(
     #           snrdb_low) + "_" + str(snrdb_high) + "_" + str(epoch) + "uni_dB.pkl"
     #      filename = filename1 + filename2
     #      torch.save(model.state_dict(), filename)