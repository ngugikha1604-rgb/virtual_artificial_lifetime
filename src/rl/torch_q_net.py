"""
torch_q_net.py — PyTorch Conv-LSTM Q-network.

The brain as a single nn.Module (sizes come from src/config.py; the local view
may be non-square, e.g. VIEW_H=6 x VIEW_W=4 when VISION_RANGE=4):

    one-hot (NUM_CELL_CLASSES, VIEW_H, VIEW_W)
        -> nn.Conv2d(NUM_CELL_CLASSES, CONV_FILTERS, 3, same) -> ReLU
        -> flatten -> CONV_OUT   -> concat[body 3] -> INPUT_SIZE
        -> nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE)  (carries its own h,c in/out)
        -> nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)   -> Q-values

Kept as its own module (not folded into the agent) so the conv+head and the
LSTM cell are separately reachable: the agent unrolls the cell over a replay
WINDOW in one autograd graph for truncated BPTT (see torch_agent.learn_windows).

A plain nn.Module also makes persistence trivial (model_io does one
torch.save / torch.load of net.state_dict()) and parameter iteration / Adam
just work. This file never touches world / agent / world_tick / run_episode —
those only call brain.act() / forward() / learn_windows() via the agent.
"""
import torch
import torch.nn as nn

import config


class ConvLSTMDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=config.NUM_CELL_CLASSES,   # one-hot channels
            out_channels=config.CONV_FILTERS,      # 8
            kernel_size=config.CONV_KERNEL,        # 3
            padding=config.CONV_PAD,               # "same": keeps VIEW_H x VIEW_W
        )
        # LSTM input = flattened conv features (CONV_OUT) + 3 body-state.
        self.lstm = nn.LSTMCell(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
        )
        self.head = nn.Linear(config.HIDDEN_SIZE, config.NUM_ACTIONS)

    def forward_obs(self, x):
        """
        x : stored observation tensor (B, OBS_SIZE) =
            [ one-hot flat (B, C*VIEW_H*VIEW_W), body (B, 3) ]  (torch.float)
        Splits it, reshapes the one-hot block back to (B, C, VIEW_H, VIEW_W)
        and runs the conv, then concatenates conv features with the body state
        -> the LSTM input (B, INPUT_SIZE).
        """
        body   = x[:, config.OBS_GRID_FLAT:]                  # (B, 3)
        onehot = x[:, :config.OBS_GRID_FLAT]                  # (B, C*VIEW_H*VIEW_W)
        B = onehot.size(0)
        oh = onehot.reshape(B, config.NUM_CELL_CLASSES,
                            config.VIEW_H, config.VIEW_W)     # (B,C,VIEW_H,VIEW_W)
        # ReLU after conv: a convolution WITHOUT a nonlinearity is just a
        # linear projection of the one-hot input, so the conv stack could not
        # learn non-linear spatial features (e.g. food-near-wall) before the
        # LSTM. Activation keeps the conv a real feature extractor.
        feat = torch.relu(self.conv(oh))  # same-pad -> (B,F,VIEW_H,VIEW_W)
        feat = feat.reshape(B, -1)        # (B, CONV_OUT)
        return torch.cat([feat, body], dim=1)                 # (B, INPUT_SIZE)

    def forward_q(self, lstm_in, h, c):
        """
        lstm_in : (B, INPUT_SIZE)    h,c : (B, HIDDEN_SIZE) each
        Returns (q (B, NUM_ACTIONS), h_next, c_next).
        """
        h_next, c_next = self.lstm(lstm_in, (h, c))
        q = self.head(h_next)
        return q, h_next, c_next
