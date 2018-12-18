from ...torch_core import *
from torch.utils.cpp_extension import load
from torch.autograd import Function

__all__ = ['QRNNLayer', 'QRNN']

import fastai
fastai_path = Path(fastai.__path__[0])/'text'/'qrnn'
files = ['forget_mult_cuda.cpp', 'forget_mult_cuda_kernel.cu']
forget_mult_cuda = load(name='forget_mult_cuda', sources=[fastai_path/f for f in files])

class ForgetMultGPU(Function):
    
    def __init__(self, batch_first:bool=True):
        self.batch_first = batch_first
        super().__init__()
    
    def forward(self, x, f, hidden_init=None):
        if self.batch_first:
            batch_size, seq_size, hidden_size = f.size()
            output = f.new(batch_size, seq_size + 1, hidden_size)
            if hidden_init is not None: output[:, 0, :] = hidden_init
            else: output = output.zero_()
        else: 
            seq_size, batch_size, hidden_size = f.size()
            output = f.new(seq_size + 1, batch_size, hidden_size)
            if hidden_init is not None: output[0, :, :] = hidden_init
            else: output = output.zero_()
        output = forget_mult_cuda.forward(x, f, output, self.batch_first)
        self.save_for_backward(x, f, hidden_init, output)
        return output[:,1:,:] if self.batch_first else output[1:,:,:]
    
    def backward(self, grad_output):
        x, f, hidden_init, output = self.saved_tensors
        grad_x, grad_f, grad_h = forget_mult_cuda.backward(x, f, output, grad_output, self.batch_first)
        return (grad_x, grad_f, grad_h) if hidden_init is not None else (grad_x, grad_f)
    
class ForgetMultCPU(nn.Module):
    
    def __init__(self, batch_first:bool=True):
        self.batch_first = batch_first
        super().__init__()
    
    def forward(self, x, f, hidden_init=None):
        result = []
        dim = (1 if self.batch_first else 0)
        forgets = f.split(1, dim=dim)
        inputs = x.split(1, dim=dim)
        prev_h = None if hidden_init is None else hidden_init.unsqueeze(dim)
        for inp, fo in zip(inputs, forgets):
            prev_h = inp * fo if prev_h is None else inp * fo + (1-fo) * prev_h
            result.append(prev_h)
        return torch.cat(result, dim=dim)

class QRNNLayer(nn.Module):
    "Apply a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence."

    def __init__(self, input_size:int, hidden_size:int=None, save_prev_x:bool=False, zoneout:float=0, window:int=1, 
                 output_gate:bool=True, use_cuda:bool=True, batch_first:bool=True):
        super().__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window,self.input_size,self.zoneout,self.save_prev_x = window,input_size,zoneout,save_prev_x
        self.hidden_size = ifnone(hidden_size, input_size)
        self.prevX = None
        self.output_gate,self.use_cuda,self.batch_first = output_gate,use_cuda,batch_first
        self.use_cuda = use_cuda
        #One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 
                                3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        if self.window == 1: source = X
        elif self.window == 2:
            Xm1 = [self.prevX if self.prevX is not None else (X[:,:1] if self.batch_first else X[:1])* 0]
            if len(X) > 1: Xm1.append((X[:,:-1] if self.batch_first else X[:-1]))
            Xm1 = torch.cat(Xm1, dim = (1 if self.batch_first else 0))
            source = torch.cat([X, Xm1], 2)
        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(*X.shape[:2], 3 * self.hidden_size)
            z_gate,f_gate,o_gate = Y.chunk(3, dim=2)
        else:
            Y = Y.view(*X.shape[:2], 2 * self.hidden_size)
            z_gate,f_gate = Y.chunk(2, dim=2)
        z_gate.tanh_()
        f_gate.sigmoid_()
        if self.zoneout and self.training:
            mask = dropout_mask(f_gate, f_gate.size(), self.zoneout).requires_grad_(False)
            f_gate = f_gate * mask
        z_gate,f_gate = z_gate.contiguous(),f_gate.contiguous()
        forget_mult = ForgetMultGPU(self.batch_first) if self.use_cuda else ForgetMultCPU(self.batch_first)
        #To avoid expected Variable but got None error
        c_gate = forget_mult(z_gate, f_gate) if hidden is None else forget_mult(z_gate, f_gate, hidden)
        output = torch.sigmoid(o_gate) * c_gate if self.output_gate else c_gate
        if self.window > 1 and self.save_prev_x: 
            self.prevX = (X[:, -1:] if self.batch_first else X[-1:]).detach()
        return output, (c_gate[:, -1:] if self.batch_first else c_gate[-1:])
    
class QRNN(nn.Module):
    "Apply a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence."

    def __init__(self, input_size:int, hidden_size:int, num_layers:int=1, bias:bool=True, batch_first:bool=True,
                 dropout:float=0, bidirectional:bool=False, **kwargs):
        assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'
        super().__init__()
        self.layers = nn.ModuleList([QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, 
                                                batch_first=batch_first, **kwargs) 
                                           for l in range(num_layers)])
        self.input_size,self.hidden_size,self.bias,self.batch_first = input_size,hidden_size,bias,batch_first
        self.num_layers,self.dropout,self.bidirectional = num_layers,dropout,bidirectional
        
    def reset(self):
        "If your convolutional window is greater than 1, you must reset at the beginning of each new sequence."
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []
        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)
            if self.dropout != 0 and i < len(self.layers) - 1:
                input = F.dropout(input, p=self.dropout, training=self.training, inplace=False)
        if self.batch_first:
            next_hidden = torch.cat(next_hidden, 1).transpose(0,1).view(self.num_layers, -1, next_hidden[0].size(-1))
        else:
            next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])
        return input, next_hidden