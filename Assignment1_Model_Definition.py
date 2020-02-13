import torch
from torch.nn import Conv1d
from torch.nn import MaxPool1d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn.functional import relu


class CnnRegressor(torch.nn.Module):
    def __init__(self, batch_size, inputs, outputs):
        super(CnnRegressor, self).__init__()
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs

        self.input_layer = Conv1d(inputs, batch_size, 1)

        self.max_pooling_layer = MaxPool1d(1)

        self.conv_layer = Conv1d(batch_size, 64, 1)

        self.flatten_layer = Flatten()

        self.linear_layer = Linear(64, batch_size)

        self.output_layer = Linear(batch_size, outputs)

    def feed(self, input):
        input = input.reshape((self.batch_size, self.inputs, 1))

        output = relu(self.input_layer(input))

        output = self.max_pooling_layer(output)

        output = relu(self.conv_layer(output))

        output = self.flatten_layer(output)

        output = self.linear_layer(output)

        output = self.output_layer(output)

        return output