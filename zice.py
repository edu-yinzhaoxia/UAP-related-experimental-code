import torch
torch.set_printoptions(precision = 32)
a = torch.tensor(4, dtype=torch.uint8)
b = a.float().div(255)
c = (b - 0.5) / 0.5
d = c * 0.5 + 0.5
e = d.mul(255)
f = e.byte()
print("a = " + str(a) + "\nb = " + str(b) + "\nc = " + str(c))
print("d = " + str(d) + "\ne = " + str(e) + "\nf = " + str(f))