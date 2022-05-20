import time
from mmdet.models import CSPDarknet
import torch

if __name__ == '__main__':
    self = CSPDarknet(deepen_factor=0.33,
                      widen_factor=0.25,
                      use_depthwise=True)
    self.eval()
    t0 = time.time()
    inputs = torch.rand(1, 3, 224, 224)
    level_outputs = self.forward(inputs)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for level_out in level_outputs:
        print(tuple(level_out.shape))
