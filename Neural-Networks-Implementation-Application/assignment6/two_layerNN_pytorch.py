import torch

class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        instantiate the layers of the network and assign them as member variables
        :param D_in: input layer dim.
        :param H: hidden layer dim.
        :param D_out: output layer dim.
        """
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear1 = torch.nn.Linear(H, D_out, bias=True)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        :param x: input data tensor
        :return: tensor of ouput data
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N is the batch-size
N, D_in, H, D_out = 64, 3, 4, 1

# load the dataset 
