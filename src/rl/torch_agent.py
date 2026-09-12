"""
torch_agent.py — PyTorch agent wrapping ConvLSTMDQN.

Phase A (done) ported the old NumPy single-step Option-B learner to torch.
Phase B replaces that learner with TRUNCATED-BPTT over replay windows so the
LSTM finally receives credit for CARRYING information across ticks instead of
only ever back-propping one detached step.

Framework boundary is drawn HERE:
  - world_tick / run_episode / visualize / live_viewer call brain.act(x,h,c),
    brain.forward(x,h,c) and brain.learn_windows(windows).
  - x, h, c arrive as NumPy arrays (stored by the episode buffer / produced by
    world_tick). TorchQAgent converts to tensors at the edge and converts
    Q-values / h_next / c_next back to NumPy so the framework-agnostic layers
    never see a torch object.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

import config


class TorchQAgent:
    def __init__(self, net, policy, learning_rate=config.LEARNING_RATE,
                 gamma=config.GAMMA, device=None,
                 target_sync_every=config.TARGET_SYNC_EVERY):
        self.net = net
        self.policy = policy
        self.gamma = gamma
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.num_actions = config.NUM_ACTIONS

        # Target network (see config.TARGET_SYNC_EVERY docstring for why): a
        # frozen copy used only to compute bootstrap targets in learn_windows,
        # re-synced from self.net every `target_sync_every` learn_windows()
        # calls. Not saved/loaded by model_io — it's always re-derived from
        # self.net (a fresh copy at construction, kept in sync afterward), so
        # a loaded checkpoint doesn't need its own target-net weights.
        self.target_net = copy.deepcopy(net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.target_sync_every = target_sync_every
        self._learn_calls = 0

    # ── conversion helpers ────────────────────────────────────────────────
    def _t(self, a):
        """NumPy array -> torch float32 vector on device, batch dim added."""
        return torch.from_numpy(np.asarray(a, dtype=np.float32)
                                ).to(self.device).unsqueeze(0)

    # ── inference (act + forward) ──────────────────────────────────────────
    def _forward_np(self, x, h, c):
        """Returns (q np(5,), h_new np(64,), c_new np(64,)). No grads."""
        with torch.no_grad():
            xt = self._t(x)
            ht = self._t(h)
            ct = self._t(c)
            lstm_in = self.net.forward_obs(xt)          # (1, INPUT_SIZE)
            q, h_new, c_new = self.net.forward_q(lstm_in, ht, ct)
        return (q.squeeze(0).cpu().numpy(),
                h_new.squeeze(0).cpu().numpy(),
                c_new.squeeze(0).cpu().numpy())

    def act(self, x, h, c):
        """Free tick: forward + epsilon-greedy sample."""
        q, h_new, c_new = self._forward_np(x, h, c)
        action = self.policy.choose_action(q, self.num_actions)
        return int(action), q, h_new, c_new

    def forward(self, x, h, c):
        """Cooldown tick: forward only (agent still 'sees' but cannot act)."""
        return self._forward_np(x, h, c)

    # ── learning (Phase B: truncated BPTT over replay windows) ────────────
    def _unroll_q(self, net, xs, h0, c0):
        """Run `net` across xs[0..L-1], carrying (h,c) forward tick by tick.
        Returns a list of per-tick Q tensors (1, NUM_ACTIONS). Grad-tracking
        follows the caller's ambient context (wrap the call in torch.no_grad()
        for a frozen rollout, e.g. the target net) — this helper itself has no
        opinion about grad, so the SAME code unrolls both the online net
        (grad ON, for the values being fit) and the target net (grad OFF, for
        the bootstrap values), which is what keeps them behaving identically
        other than which weights and which grad mode they run under."""
        qs = []
        h, c = h0, c0
        for xk in xs:
            qk, h, c = net.forward_q(net.forward_obs(xk), h, c)
            qs.append(qk)
        return qs

    def learn_windows(self, windows):
        """
        windows: list of EPISODE WINDOWS from the buffer. Each is a contiguous
        slice of one episode, length L <= WINDOW_N, of tick-records
             (x, h0, c0, action:int, reward:float, done:bool)   # NumPy
        (h0, c0) is the LSTM state the net ACTUALLY had before that tick (the
        hidden the previous tick's live forward produced). A window that starts
        mid-episode therefore still has an authentic recurrent context, and a
        window that does not reach the episode's done tick simply ends at a
        real continuing state — never a false "world ended here".

        TRUNCATED BPTT over one combined update, WITH a target network:
          - Unroll the ONLINE net across every window's L observations in ONE
            graph (grad ON), carrying (h,c) with autograd so each step's
            Q-gradient reaches back through all earlier LSTM recurrences
            (multi-step credit for memory).
          - Separately unroll the TARGET net (a periodically-frozen copy, see
            config.TARGET_SYNC_EVERY) across the SAME xs, starting from the
            SAME (h0, c0) — i.e. the target net gets its own honest
            hidden-state rollout, not a value borrowed from the online unroll.
            This is the standard DQN target-network fix (decouple "the value
            being updated" from "the value the update chases") ported to the
            windowed/recurrent setting; without it the bootstrap target moves
            every single Adam step, which is a known source of oscillation.
          - target_k = r_k + gamma * max_a' q_target_{k+1}   [target net, no grad]
          - The final step of a window is supervised ONLY when the buffer says
            the episode actually ended there (done) — using the ONLINE net's
            own q_last (a true terminal reward needs no bootstrap, so there is
            no reason to prefer the target net there); otherwise that tail Q is
            produced (so its gradient flows backward to help earlier targets)
            but receives no self-derived bootstrap — avoiding the "fake
            terminal at a truncated window boundary" bias.
        Losses from all windows are summed and ONE Adam step is taken, then the
        target net is re-synced from the online net every `target_sync_every`
        calls to this method.
        """
        q_parts, t_parts = [], []

        for win in windows:
            L = len(win)
            if L < 2:
                continue

            xs      = [self._t(w[0]) for w in win]        # (1, OBS_SIZE) each
            actions = [w[3] for w in win]
            rewards = [float(w[4]) for w in win]
            dones   = [bool(w[5]) for w in win]
            h0 = self._t(win[0][1])
            c0 = self._t(win[0][2])

            # online unroll (grad ON) — values being fit + terminal estimate
            qs = self._unroll_q(self.net, xs, h0, c0)

            # target unroll (grad OFF, frozen weights) — bootstrap values only
            with torch.no_grad():
                qs_target = self._unroll_q(self.target_net, xs, h0, c0)

            for k in range(L - 1):
                q_k  = qs[k][0, actions[k]]                 # keeps the BPTT graph
                q_nx = qs_target[k + 1]                     # frozen target net
                boot = rewards[k] + self.gamma * float(q_nx.max())
                q_parts.append(q_k)
                t_parts.append(boot)

            # supervise the actual terminal tick if the window reached it
            if dones[L - 1]:
                q_last = qs[L - 1][0, actions[L - 1]]
                q_parts.append(q_last)
                t_parts.append(rewards[L - 1])

        if not q_parts:
            return 0.0

        qq = torch.stack(q_parts)
        tt = torch.tensor(t_parts, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        loss = F.mse_loss(qq, tt)
        loss.backward()
        self.optimizer.step()

        self._learn_calls += 1
        if self._learn_calls % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return float(loss.item())
