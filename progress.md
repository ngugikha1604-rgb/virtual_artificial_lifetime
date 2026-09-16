# Progress Log — Virtual Lifetime

> File này tóm tắt toàn bộ những gì đã xây dựng, tại sao làm vậy, và những bài học rút ra.
> Đọc file này đầu mỗi session mới để không mất ngữ cảnh.

---

## 🎯 Mục đích & kết quả mong đợi của các thay đổi gần đây

> Mục này để Khanh đối chiếu xem Claude có hiểu đúng ý không — tập trung vào **tại sao** làm và
> **kỳ vọng quan sát được gì** nếu đúng, không đi sâu "dùng công cụ/dòng code nào" (phần đó nằm
> ở các mục kỹ thuật phía dưới, dùng khi cần debug).

**1. Target network cho việc học (`learn_windows`)**
- Mục đích: tránh việc network vừa là "người học" vừa là "thước đo để tự chấm điểm mình" cùng
  lúc — dễ gây dao động/không ổn định khi train lâu.
- Kết quả mong đợi: quá trình học ổn định hơn qua thời gian, loss/hành vi không bị "giật"
  hoặc dao động thất thường khi train đủ lâu.

**2. Reward shaping chỉ tính theo cái agent THẤY ĐƯỢC (không còn biết vị trí food/hazard ẩn)**
- Mục đích: trước đó agent được "mớm" thông tin nó không thể biết (food ở đâu dù không nhìn
  thấy) — giống như gian lận, và tạo tín hiệu học mâu thuẫn/nhiễu.
- Kết quả mong đợi: hành vi agent phản ánh đúng những gì nó thực sự cảm nhận được qua tầm
  nhìn — có thể học chậm hơn lúc đầu (vì bớt được "mớm bài"), nhưng hành vi cuối cùng đáng tin
  hơn để quan sát/diễn giải.

**3. Epsilon được nhớ lại giữa các lần chạy + cờ `--reset-epsilon`**
- Mục đích: Khanh chia nhỏ việc train thành nhiều lần chạy (vì 1 lần chạy đủ dài rất lâu) —
  không muốn mỗi lần chạy lại agent bị "bắt đầu khám phá lại từ đầu" một cách vô lý.
- Kết quả mong đợi: dù chạy `run_experiment.py` bao nhiêu lần, tách rời hay liên tục, quá trình
  giảm dần độ ngẫu nhiên của agent luôn tiếp diễn mượt như thể đó là MỘT lần chạy dài duy nhất.
  Khi nào muốn cho agent "tò mò" lại từ đầu mà không mất bộ não đã học, dùng `--reset-epsilon`.

**4. Phạt khi bị đói (starvation), tương xứng với phạt khi dính hazard**
- Mục đích: đói lâu và dính hazard đều dẫn đến chết như nhau, nhưng trước đó agent chỉ "cảm nhận
  rõ" hazard là nguy hiểm (có phạt trực tiếp), còn đói thì gần như vô hình về mặt tín hiệu học.
- Kết quả mong đợi: theo thời gian, agent chủ động tìm ăn trước khi cạn năng lượng, không chỉ né
  hazard — nhìn hành vi tổng thể sẽ thấy "biết lo cho bản thân" toàn diện hơn, không lệch hẳn
  về một loại nguy hiểm.

**5. Có bản backup / checkpoint định kỳ trong lúc train, không đợi train xong mới lưu**
- Mục đích: (a) tránh mất hết tiến độ nếu máy/tiến trình bị dừng giữa chừng, (b) tránh việc bộ
  não cuối cùng lưu lại chỉ là "bất kỳ lúc nào training dừng" — muốn có điểm neo tốt để so sánh.
- Kết quả mong đợi: có thể yên tâm chạy training dài mà không sợ mất trắng nếu gián đoạn; có
  cơ sở (in ra khi có "best" mới) để biết training có đang cải thiện thật hay không, không phải
  đoán qua vài dòng log cuối cùng.

**6. Thế giới có địa hình: đất (soil), cỏ (grass), tường (wall), nước (water)**
- Mục đích: theo đúng yêu cầu "thế giới hợp lý hơn" — world trước đó chỉ là ô trống rải vài vật
  thể, không có cảm giác một không gian thật để di chuyển trong đó.
- Kết quả mong đợi: nhìn vào GIF/live_viewer sẽ thấy một world có hình dạng/texture rõ ràng chứ
  không còn trống trơn; quan trọng hơn, có thêm một loại quyết định mới để quan sát hành vi phát
  triển: agent có học được lúc nào đáng băng qua nước (tốn thời gian/sức) để đi tắt, và lúc nào
  nên đi vòng tránh, hay không? Đây là kiểu trade-off hành vi đúng tinh thần "xem hành vi phát
  triển" mà dự án đang nhắm tới.

**7. Gộp về đúng 1 file "bộ não" duy nhất (`best_model.pt`)**
- Mục đích: trước đó có 2 file khác nhau cho 2 công cụ (train vs xem trực tiếp) — rườm rà, dễ
  nhầm "đang xem/train đúng bộ não nào".
- Kết quả mong đợi: bất kể dùng công cụ nào (train bằng lệnh, hay mở lên xem trực tiếp), luôn
  chắc chắn đang thao tác trên cùng MỘT bộ não duy nhất, tiến độ không bị phân mảnh giữa các
  công cụ.

---

## ⚠️ THAY ĐỔI LỚN (2026-09): đổi mục tiêu dự án + refactor toàn bộ sang PyTorch

Hai thay đổi nền tảng vừa xảy ra, cả hai đều **ghi đè lên các nguyên tắc cũ** mô tả ở phần dưới của
file này (phần dưới giữ lại vì lịch sử/lý do thiết kế vẫn còn giá trị tham khảo, nhưng không còn
là luật hiện hành nữa):

1. **Mục tiêu dự án đã đổi.** Trước đây mục tiêu là *học sâu về cơ chế neural network* bằng cách tự
   tay cài đặt mọi thứ (forward/backward, LSTM, Conv2D, optimizer...) từ NumPy thuần, và cái quan
   tâm chính là *weight thay đổi thế nào qua thời gian*. Giờ mục tiêu là **xây một simulation world**
   — có thể dùng cho game, có thể chỉ để quan sát/hình dung cách một sinh vật 2D sống, tìm ăn, tránh
   nguy hiểm, phát triển theo thời gian. Cái quan tâm chính bây giờ là **behavior thay đổi thế nào**,
   không phải weight. Việc "tự tay implement từ scratch để hiểu" không còn là ràng buộc bắt buộc.
2. **Toàn bộ neural net brain đã refactor sang PyTorch.** Các file NumPy-thuần cũ trong `src/brain/`
   (`layer.py`, `lstm_cell.py`, `conv2d.py`, `_conv_gradcheck.py`) và agent NumPy cũ trong `src/rl/`
   (`lstm_agent.py`) **không còn tồn tại trong codebase** — đã bị thay thế, không chỉ archive. `src/brain/`
   giờ chỉ còn `model_io.py` (save/load `state_dict` của PyTorch). Backprop, Adam, LSTM cell, Conv2D
   giờ đều là code của `torch`, không phải code tự viết.

Vì mục tiêu học "hiểu từng dòng gradient" không còn là ưu tiên, dùng framework để tập trung công sức
vào world/behavior/observability hợp lý hơn. **README.md đã được viết lại theo tinh thần mới** — đọc
mục 1–5 ở đó trước khi đề xuất bất kỳ thay đổi kiến trúc nào liên quan đến "phải tự cài đặt".

### Trạng thái lỗi hiện tại (CẦN THEO DÕI)

Khanh báo là refactor này **vẫn còn lỗi**, chưa nói rõ lỗi gì. Bản thân codebase đọc tĩnh (static
review) không phát hiện lỗi cú pháp/logic rõ ràng trong `torch_q_net.py`, `torch_agent.py`,
`lstm_replay_buffer.py`, `world_tick.py`, `run_episode.py`, `live_viewer.py`, `visualize.py`,
`model_io.py` — nhưng chưa được **chạy thử thật** (môi trường Claude hiện tại không có quyền chạy
Python trên máy Windows của Khanh, và sandbox Linux của Claude không có `torch`/`pygame` cài sẵn).
**Bước tiếp theo bắt buộc**: Khanh dán traceback lỗi cụ thể (từ `run_experiment.py`, `run_episode.py`,
hoặc `live_viewer.py`) vào session tiếp theo để Claude debug đúng chỗ, thay vì đoán.

**Cập nhật — đã thử chạy thật trong sandbox (Linux, CPU, torch 2.14, pygame 2.6.1, matplotlib
3.10.8) và KHÔNG tái hiện được lỗi nào** ở các đường chạy sau:
- `run_episode.main(num_lifetimes=15)` — training loop trần, kích hoạt cả `learn_windows` (buffer
  vượt `MIN_EPISODES=10`) — chạy sạch.
- `run_experiment.py --lifetimes 15` (brain mới) — train + save `.pt` + render GIF + ghi CSV — chạy
  sạch, output số liệu hợp lý (food/reward tăng nhẹ qua 15 lifetime).
- `run_experiment.py --lifetimes 5` chạy lần 2 (load `.pt` vừa lưu để train tiếp) — chạy sạch, xác
  nhận `model_io.load_lstm_weights` hoạt động đúng với checkpoint torch thật.
- `run_experiment.py --no-train` (load `.pt`, demo episode + GIF, không ghi đè CSV) — chạy sạch.
- `live_viewer.py` chạy headless (`SDL_VIDEODRIVER=dummy`), bật training, 600 tick liên tục qua
  nhiều lifetime — chạy sạch, không lỗi, không crash khi agent chết/reset giữa chừng.
- 80 lifetime liên tục kiểm tra `torch.isnan()` trên toàn bộ parameters mỗi 10 lifetime — **không
  có NaN/inf xuất hiện** ở bất kỳ điểm kiểm tra nào.
- Ghi nhận tốc độ: ~1.2s/lifetime trung bình sau khi `learn_windows` bắt đầu chạy (buffer đủ
  `MIN_EPISODES`) trên CPU sandbox — với `NUM_LIFETIMES=4000` mặc định, một lần chạy đầy đủ
  `run_experiment.py` sẽ mất khoảng hơn 1 giờ trên phần cứng tương đương. Đây không phải lỗi,
  nhưng đáng lưu ý nếu Khanh thấy chương trình "treo" khi chạy full — nhiều khả năng chỉ là đang
  chạy chậm chứ không crash (đối chiếu bằng `PRINT_EVERY=100` — nếu vẫn thấy dòng log mới xuất
  hiện định kỳ thì không phải bug, chỉ là chờ lâu).

**Kết luận tạm thời**: lỗi Khanh gặp nhiều khả năng KHÔNG nằm ở luồng chạy "vanilla" mà 5 kịch bản
trên bao phủ. Các khả năng còn lại, theo thứ tự nghi ngờ: (1) đặc thù Windows — ví dụ font
`"segoeui"` trong `live_viewer.py` chỉ có trên Windows, có thể lỗi/fallback khác trên máy Khanh dù
ở Linux sandbox thì Pygame fallback êm; (2) tương tác bàn phím thực tế trong `live_viewer.py`
(phím R reset giữa lúc đang bận cooldown, phím S lưu weight, v.v.) — sandbox chỉ test được
`step_simulation()` gọi trực tiếp, chưa test qua vòng lặp sự kiện Pygame thật; (3) có GPU/CUDA
trên máy Khanh và lỗi chỉ xảy ra trên CUDA path (`torch.cuda.is_available()` → device khác); (4) lỗi
xảy ra ở lifetime rất xa (hàng trăm/nghìn) mà 80-lifetime test chưa chạm tới; (5) lỗi không phải
exception mà là **lỗi hành vi/logic** (VD agent vẫn "đứng yên"/"quay vòng" dù không crash) — nếu
đúng vậy thì đây không phải bug kỹ thuật, mà là vấn đề tinh chỉnh reward/hyperparameter giống các
lần trước.

→ **Cần Khanh cho biết: lỗi cụ thể là gì (traceback, hay hành vi bất thường), chạy lệnh gì, và có
GPU/CUDA trên máy không** — để không đoán mò tiếp.

### Đã sửa thêm (2026-09, sau khi đối chiếu review của DeepSeek — 5/5 điểm DeepSeek nêu hoá ra
### KHÔNG áp dụng cho code hiện tại, xem hội thoại; nhưng nhân dịp review lại, đã tự sửa 2 điểm
### thiết kế thật sự đáng sửa mà Claude tìm thấy khi đọc kỹ:

1. **Thêm target network cho `learn_windows`** (`src/rl/torch_agent.py`). Trước đó bootstrap target
   lấy từ chính unroll online (`qs[k+1].detach()`) — đúng về mặt kỹ thuật (không stale-state drift
   như style R2D2 cũ), nhưng vẫn là kiểu "chasing a moving target" kinh điển vì target luôn bằng
   đúng weight hiện tại. Giờ `TorchQAgent` giữ thêm `self.target_net` (bản copy, `requires_grad=False`),
   unroll riêng (không gradient) trên CÙNG `(h0, c0)` bắt đầu window để lấy bootstrap value, đồng bộ
   lại từ online net mỗi `config.TARGET_SYNC_EVERY=200` lần gọi `learn_windows` (~4-5 lifetime/lần).
   Không cần đổi `model_io`/save-load — target net luôn được suy ra lại từ checkpoint online khi
   khởi tạo `TorchQAgent`, không tự lưu riêng. Đã verify bằng script test: target net theo sát
   online net, lệch nhỏ dần giữa các lần sync (test với `target_sync_every=5`, max diff ~0.003 sau
   374 learn call).
2. **Reward shaping giờ chỉ tính trên entity NẰM TRONG TẦM NHÌN** (`world.visible_entity_positions()`
   mới trong `src/world/world.py`, dùng trong `compute_reward()` ở `run_episode.py`). Trước đó
   `compute_reward` dùng `world.food_positions`/`world.hazard_positions` — TOÀN BỘ vị trí thật
   trong world, kể cả những entity nằm ngoài local view. Về lý thuyết, chứng minh "potential-based
   shaping additive-safe" (Ng et al. 1999) chỉ đúng dưới full observability; world này là POMDP
   (agent chỉ thấy local view), nên shaping dựa trên vị trí không quan sát được là tín hiệu **rò rỉ
   thông tin toàn tri + nhiễu không học được** (hai tick có local view giống hệt nhau vẫn nhận
   shaping khác nhau tuỳ food/hazard ẩn ở đâu) — nghi là một nguyên nhân góp phần vào hành vi
   "đứng yên"/"spinning" quan sát được trước đây, chưa từng được ghi nhận trong danh sách nguyên
   nhân cũ (SURVIVAL_BONUS, EPSILON_MIN, single-step BPTT, vision range).
   - `world_tick.TickResult` giờ có thêm field `prev_facing` (facing TRƯỚC tick, vì `agent.facing`
     đã bị `world.step()` cập nhật sang facing mới khi caller nhận được `TickResult`) để
     `compute_reward` biết chính xác agent đã nhìn thấy gì lúc ra quyết định.
   - `compute_reward()` đổi chữ ký: `compute_reward(event, prev_pos, prev_facing, new_pos, world)`
     — cả `run_episode.py` và `live_viewer.py` đã cập nhật chỗ gọi.

**Đã verify cả 2 thay đổi bằng chạy thật trong sandbox** (không chỉ đọc code): `run_episode.main()`,
`run_experiment.py --lifetimes 15` (brain mới), `run_experiment.py --lifetimes 3` (load lại để train
tiếp), và `live_viewer.py` headless 400 tick với training bật — tất cả chạy sạch, không exception,
số liệu food/reward/steps hợp lý. **Roadmap mục 1 ("optional target network") trong `torch_agent.py`
docstring cũ giờ đã DONE.**

⚠️ Hai thay đổi này KHÔNG phải là fix cho lỗi bí ẩn Khanh từng báo (vẫn chưa xác định được lỗi đó là
gì — xem mục ngay phía trên). Đây là cải thiện chất lượng training riêng biệt, tìm ra khi Claude đọc
lại kỹ kiến trúc theo yêu cầu "xem codebase ổn chưa". Baseline số liệu ở bảng "Kết quả thực nghiệm"
bên dưới vẫn CHƯA phản ánh 2 thay đổi này — cần train lại (4000 lifetime) để có số liệu mới.

`results/` hiện đã CÓ artifact thật (khác với ghi chú "rỗng" ở các mục cũ bên dưới — ghi chú đó lỗi thời):
`trained_brain_conv_lstm.pt`, `phase6_multi_entity.csv`, `lifetime_demo_lstm.gif` đều tồn tại, nghĩa
là ít nhất một lần chạy `run_experiment.py` đã hoàn tất tới bước lưu file — nhưng không rõ đó là kết
quả của bản đã hết lỗi hay của một lần chạy trước khi lỗi hiện tại xuất hiện.

---

## Kiến trúc hiện tại (PyTorch Conv-LSTM DQN, Phase B truncated-BPTT)

```text
World (grid, mặc định 10×10, xem src/config.py)
  ↓  get_local_view(position, facing)
  ↓  local view (VIEW_H × VIEW_W, mặc định 6×4), mỗi cell code int 0..5
  ↓
encode_observation() [NumPy, framework-agnostic]
  ↓  one-hot (NUM_CELL_CLASSES=6, VIEW_H, VIEW_W) flatten + body-state(3)
  ↓  = obs vector OBS_SIZE (mặc định 147), đây là cái được lưu trong replay
  ↓
ConvLSTMDQN (torch.nn.Module, src/rl/torch_q_net.py)
  ├─ nn.Conv2d(NUM_CELL_CLASSES → CONV_FILTERS, k=3, same-pad) + ReLU
  ├─ flatten conv feature + concat body(3)  → INPUT_SIZE
  ├─ nn.LSTMCell(INPUT_SIZE → HIDDEN_SIZE=64)
  └─ nn.Linear(HIDDEN_SIZE → NUM_ACTIONS=5)  → Q-values
  ↓
TorchQAgent (src/rl/torch_agent.py)
  - act()/forward(): numpy in → torch (no_grad) → numpy out (biên framework nằm ở đây)
  - learn_windows(windows): Phase B — truncated BPTT thật sự qua nhiều tick
    (không còn single-step Option-B như bản NumPy cũ). Unroll LSTM qua một
    window tick liên tiếp trong MỘT graph autograd, bootstrap target lấy từ
    q(k+1) của chính unroll đó (detached), Adam step 1 lần / batch windows.
  ↓
EpsilonGreedyPolicy (src/rl/policy.py) — không đổi nhiều so với bản cũ
  ↓
World.step(action, position, facing) → (new_pos, new_facing, event)
  ↓
Agent.apply_energy_cost / apply_event / advance_age (src/world/agent.py)
  ↓
LSTMReplayBuffer (src/rl/lstm_replay_buffer.py)
  - lưu NGUYÊN EPISODE (không phải transition rời rạc), mỗi tick giữ luôn
    (x, h0, c0, action, reward, done) — h0/c0 là hidden TRƯỚC tick đó, để
    một window bất kỳ giữa episode vẫn unroll đúng ngữ cảnh.
  - sample_windows(k, window_n): lấy k window ngẫu nhiên (≤ WINDOW_N tick)
    từ các episode đã đóng, dùng cho learn_windows().
```

`world_tick.py` vẫn là nơi duy nhất chứa logic "1 tick vật lý" dùng chung cho
`run_episode.py` (training), `visualize.py` (ghi GIF), `live_viewer.py` (Pygame
interactive) — nguyên tắc này (tránh copy-paste 3 nơi) vẫn còn hợp lý và giữ nguyên.

### Vì sao đổi từ single-step Option-B (NumPy) sang windowed truncated-BPTT (torch)

Bản NumPy cũ chỉ backprop **một bước** mỗi lần học — LSTM không bao giờ thực sự học cách
*giữ* thông tin qua nhiều tick, vì gradient không bao giờ chảy ngược qua nhiều bước thời
gian. `TorchQAgent.learn_windows()` unroll cả một đoạn liên tiếp (tối đa `WINDOW_N` tick)
trong một đồ thị autograd duy nhất, nên LSTM cell mới thật sự có tín hiệu gradient khuyến
khích "nhớ" điều gì đó qua thời gian (ví dụ: vị trí food đã thấy nhưng hiện không còn trong
tầm nhìn). Đây là lý do chính đáng cho việc chuyển sang torch, ngoài lý do mục tiêu dự án
đổi hướng.

---

## Layer 1: Brain (`src/brain/` + `src/rl/torch_*.py`)

**Không còn NumPy thuần.** `src/brain/` giờ chỉ có:

### `model_io.py`
- `save_lstm_weights(net, path)` / `load_lstm_weights(net, path)` — wrap `torch.save`/`torch.load`
  trên `net.state_dict()`. File lưu là `.pt`, không phải `.npz` nữa.
- `load_lstm_weights` vẫn giữ nguyên tinh thần cũ: nếu shape/key không khớp kiến trúc hiện tại
  (đổi `VISION_RANGE`, đổi `HIDDEN_SIZE`, v.v.) thì raise `ValueError` thay vì load sai — caller
  (`run_experiment.py`, `live_viewer.py`) bắt lỗi này và khởi tạo lại brain mới thay vì crash.

### `src/rl/torch_q_net.py` — `ConvLSTMDQN(nn.Module)`
- Kiến trúc: `nn.Conv2d` → ReLU → flatten → concat body(3) → `nn.LSTMCell` → `nn.Linear`.
- Tách riêng `forward_obs(x)` (conv + flatten + concat body) và `forward_q(lstm_in, h, c)`
  (LSTM step + head) để agent có thể unroll `forward_q` nhiều lần trong 1 window mà chỉ gọi
  `forward_obs` một lần mỗi tick.

### `src/rl/torch_agent.py` — `TorchQAgent`
- Biên NumPy↔torch nằm chính xác ở đây: mọi thứ phía world/agent/world_tick/replay buffer chỉ
  thấy NumPy array; `TorchQAgent` là nơi duy nhất `.to(device)`, tạo tensor, gọi `.backward()`.
- `act(x, h, c)`: forward không gradient + epsilon-greedy sample → dùng cho tick "tự do" (agent
  được chọn action mới).
- `forward(x, h, c)`: forward không gradient, không sample action → dùng cho tick "cooldown"
  (agent đang bận thực hiện action cũ, world_tick vẫn cho brain "nhìn" nhưng ép action=STAY).
- `learn_windows(windows)`: xem giải thích ở trên — truncated BPTT, 1 Adam step cho toàn bộ batch
  windows gộp lại (loss = MSE tổng hợp qua mọi window).

---

## Layer 2: World (`src/world/`) — không đổi kiến trúc so với trước khi refactor torch

### `world.py` — World (grid vuông, mặc định 10×10, entity system tổng quát)
- Entity kiểu tổng quát: `entities: list[{"type", "pos"}]`, khai báo qua `ENTITY_SPECS`
  (`food_low`, `food_high`, `hazard`). Thêm loại vật mới chỉ cần thêm 1 entry, không sửa
  grid/step logic.
- `get_local_view(position, facing)`: trả về local view `(VIEW_H, VIEW_W)` xoay theo hướng nhìn
  (mặc định 6×4 = `VISION_RANGE(4) + BEHIND_ROWS(2)` hàng × `VISION_WIDTH(4)` cột — kích thước
  đọc từ `config.py`, không hard-code).
- `step(action, position, facing)`: trả `(new_pos, new_facing, event)`. Food ăn xong respawn
  ngay ở ô khác; hazard bắn event mỗi tick agent đứng trên đó (không bị tiêu thụ).

### `agent.py` — Agent (body mechanics)
- `internal_state = [energy/MAX, health/MAX, age/max_age]` (3 giá trị, nối vào sau one-hot
  trong observation vector).
- 3 method `apply_energy_cost` / `apply_event` / `advance_age` gộp toàn bộ logic "một tick body"
  — dùng chung bởi `world_tick.py`, không còn copy-paste ở 3 nơi.
- `ACTION_DURATION = {0:1, 1:3, 2:3, 3:1, 4:1}` — di chuyển tiến/lùi tốn 3 tick "bận" (cooldown),
  đứng yên/xoay là tức thời. Đây là lý do `world_tick` có nhánh `is_free` / cooldown.

---

## Layer 3: RL driver files (`src/rl/`)

### `lstm_q_network.py`
- Hai trách nhiệm tách bạch rõ ràng: (1) observation builder **NumPy thuần**
  (`encode_observation`/`build_observation`, `zero_state()`) dùng chung bởi mọi driver, không
  đụng torch; (2) `build_lstm_network()` — factory tạo `ConvLSTMDQN` (torch), import torch chỉ ở
  đây.

### `lstm_replay_buffer.py` — `LSTMReplayBuffer`
- Lưu **cả episode** (không phải transition rời rạc) — xem giải thích Phase B ở trên.
- `REPLAY_CAPACITY` giờ đếm **số episode**, không phải số transition (khác bản NumPy cũ đếm
  transition — đổi đơn vị vì lý do bộ nhớ, xem comment trong `config.py`).

### `policy.py` — `EpsilonGreedyPolicy`
- Không đổi nhiều: `choose_action`, `decay()` (vẫn phải gọi đúng 1 lần/lifetime, không phải mỗi
  tick — bug cũ đã từng mắc phải, `live_viewer.py` có comment nhắc lại điều này tường minh).

### `world_tick.py`
- Không đổi nguyên tắc so với bản NumPy: 1 hàm `world_tick()` dùng chung cho
  `run_episode.py`/`visualize.py`/`live_viewer.py`, giữ "vật lý" tách khỏi "chính sách" (reward
  shaping, đếm counter, vẽ hình là việc riêng của từng caller).

### `run_episode.py`
- `compute_reward()`: potential-based shaping cho food (kéo lại gần) và hazard (đẩy ra xa) +
  bonus cố định khi ăn + phạt cố định khi đứng trên hazard — logic reward giữ nguyên từ bản
  NumPy, chỉ có brain phía dưới đổi.
- `run_episode()`: vòng lặp 1 lifetime, gọi `world_tick`, đẩy vào `replay.push(...)`, gọi
  `brain.learn_windows(replay.sample_windows(...))` mỗi `LEARN_EVERY` tick (không phải mỗi tick).
  Hỗ trợ `record=True` để lấy `frames` cho GIF từ ĐÚNG lifetime training thật, tránh bug cũ (GIF
  từng được ghi từ một lifetime riêng chưa từng nằm trong CSV).

### `visualize.py`
- `snapshot()` + `save_lifetime_gif()` — không đổi so với bản NumPy, vẽ bằng matplotlib.

### `live_viewer.py`
- Pygame interactive viewer, gọi `world_tick()` dùng chung. Phím `T` bật/tắt online-training
  trực tiếp trong lifetime đang xem; `policy.decay()` gọi đúng 1 lần/lifetime (khi agent chết),
  không phải mỗi tick. Phím `S` lưu weight thủ công; auto-save khi thoát nếu training đang bật.

### `experiments/phase5_online_qlearning/run_experiment.py`
- Entry point train + save + demo GIF + CSV trong 1 lệnh. Load `.pt` cũ để train tiếp nếu có
  (bắt `ValueError` từ `model_io` nếu kiến trúc đổi → tạo brain mới). `--no-train` chỉ demo bằng
  weight đã có, không train, không ghi đè CSV cũ.

---

## Cấu trúc thư mục hiện tại (đã xác nhận bằng cách đọc trực tiếp filesystem)

```text
virtual_lifetime/
├── progress.md
├── README.md
├── src/
│   ├── config.py           Nguồn chân lý duy nhất cho mọi hằng số (world/RL/reward/agent)
│   ├── brain/
│   │   └── model_io.py     Save/Load state_dict() của ConvLSTMDQN (torch)
│   ├── world/
│   │   ├── world.py
│   │   └── agent.py
│   └── rl/
│       ├── torch_q_net.py        ConvLSTMDQN (nn.Module: Conv2d+ReLU, LSTMCell, Linear)
│       ├── torch_agent.py        TorchQAgent (act/forward/learn_windows, biên numpy↔torch)
│       ├── lstm_q_network.py     encode_observation/build_observation, zero_state, factory
│       ├── lstm_replay_buffer.py Episode buffer + sample_windows (Phase B)
│       ├── policy.py             EpsilonGreedyPolicy
│       ├── world_tick.py         Tick vật lý dùng chung
│       ├── run_episode.py        Training loop 1 lifetime
│       ├── visualize.py          Matplotlib GIF recorder
│       └── live_viewer.py        Pygame interactive 2D viewer
├── experiments/
│   └── phase5_online_qlearning/
│       └── run_experiment.py     Train + save + demo GIF + CSV, entry point chính
└── results/
    ├── trained_brain_conv_lstm.pt
    ├── phase6_multi_entity.csv
    └── lifetime_demo_lstm.gif
```

> Không còn thư mục `src/_legacy/` — các file NumPy Phase 1-5 cũ (`network.py`, `optimizer.py`,
> `loss.py`, `activation.py`, `q_network.py`, `q_agent.py`, `frame_buffer.py`, `replay_buffer.py`,
> `layer.py`, `lstm_cell.py`, `conv2d.py`, `lstm_agent.py`) đã bị xoá hẳn khỏi repo trong đợt
> refactor torch này (không chỉ archive như đợt dọn dẹp trước).

---

## Kết quả thực nghiệm trước đây (bản NumPy, tham khảo lịch sử — CHƯA có baseline mới bằng torch)

| Setup | Architecture | first 100 food/lt | last 100 food/lt |
|---|---|---|---|
| Ban đầu (dx, dy) | MLP no replay | 0.32 | 10.42 |
| Local view | MLP no memory | ~0.4 | ~0.2 (degraded) |
| Local view | Frame Stacking (N=4, 66-D) + Replay 20k | 0.82 | 4.10 |
| Local view + LSTM | LSTM (Option-B, 19-D) + Replay 20k | 0.85 | ~4.5+ |
| + food_low/food_high/hazard | LSTM (Option-B, 19-D) + Replay 20k | 2.06 food/lt, 155 steps/lt | 5.84 food/lt, ~194 steps/lt |
| + one-hot + Conv2D (NumPy) | Conv-LSTM (Option-B) | *(chưa train đủ trước khi đổi sang torch)* | — |

**Chưa có con số benchmark cho bản torch Phase B (windowed BPTT).** Cần chạy hết lỗi hiện tại rồi
train lại (`run_experiment.py`, mặc định `NUM_LIFETIMES=4000` trong `config.py`) để có baseline
mới, đặc biệt vì cơ chế học (windowed BPTT) khác hẳn Option-B single-step nên số cũ không so sánh
trực tiếp được nữa.

---

## Bài học thiết kế vẫn còn giá trị (giữ nguyên qua đợt refactor torch)

- **Equal action energy costs** ngăn "đứng yên" trở thành local optimum.
- **Potential-based reward shaping** (Ng et al. 1999, additive-safe) cho gradient signal khi
  food/hazard ở xa tầm với.
- **`policy.decay()` phải gọi đúng 1 lần/lifetime**, không phải mỗi tick — gọi mỗi tick làm epsilon
  sụp đổ gần như ngay lập tức. Bug này từng xảy ra ở `live_viewer.py`, đã fix và giữ comment cảnh báo.
- **Demo recording cần epsilon > 0** (VD 0.1) để agent chưa train kỹ không bị kẹt vòng lặp
  deterministic.
- **Hazard/food constants (`HAZARD_DAMAGE`, `FOOD_BONUS`, ...) đều là giá trị chọn tạm** — chưa
  qua thực nghiệm tinh chỉnh dài hạn, xem `config.py`.
- **Cell one-hot multi-channel > scalar encoding**: tránh gán ngầm "tốt/xấu" cứng lên vật thể rời
  rạc — network tự học tầm quan trọng từng kênh.

---

## Roadmap tiếp theo (đã cập nhật theo mục tiêu mới: simulation/behavior, không phải weight-learning)

1. **CẤP BÁCH — fix lỗi torch refactor.** Cần traceback cụ thể từ Khanh. Không đoán mò kiến trúc
   nữa; review tĩnh không thấy lỗi rõ ràng nên nhiều khả năng là runtime (shape mismatch lúc chạy,
   thiếu package, lỗi Pygame trên máy có màn hình, hoặc lỗi trong vòng lặp windows rỗng khi buffer
   chưa đủ episode).
2. Train lại để có baseline torch/windowed-BPTT thật (thay bảng kết quả NumPy cũ ở trên).
3. **Trọng tâm mới — quan sát & trực quan hoá behavior**, không phải chỉnh weight:
   - Cải thiện `live_viewer.py` / GIF export để dễ "xem" một sinh vật sống một đời — có thể thêm
     replay tua nhanh/chậm, so sánh hành vi giữa các giai đoạn train (đầu đời train vs cuối đời train).
   - Cân nhắc thêm log hành vi định tính (không chỉ số liệu food/hazard) — VD heatmap vị trí, biểu
     đồ hành động theo thời gian trong 1 lifetime — để "nhìn thấy" sự phát triển hành vi rõ hơn.
4. Thế giới phong phú hơn (world lớn hơn, địa hình, có thể thêm ngày/đêm) — chỉ thêm khi phục vụ
   trực tiếp mục tiêu "quan sát behavior" hoặc "làm game", không thêm vì "cho đủ phase" nữa.
5. Multi-agent — cần refactor lớn (`World.step()` hiện chỉ nhận 1 agent/lần gọi). Ưu tiên thấp hơn
   việc fix lỗi + có behavior quan sát được ổn định trước.
6. Nếu mục tiêu game rõ hơn theo thời gian: cân nhắc tách phần "brain training" ra khỏi phần
   "world rendering/game loop" rõ ràng hơn nữa, để có thể đóng gói world như một sản phẩm xem/chơi
   độc lập với việc có đang train hay không.

---

## Đã sửa thêm (2026-09, sau buổi review "codebase ổn chưa") — 4 điểm (a)/(b)/(d) + checkpointing

Sau khi review chi tiết toàn bộ codebase, phát hiện thêm 4 vấn đề (a-d, xem hội thoại đầy
đủ nếu cần) và đã xử lý theo yêu cầu cụ thể của Khanh:

**(a) + (b) — Epsilon không được persist qua các lần chạy `run_experiment.py`.**
Không đổng đến `EPSILON_MIN`/`EPSILON_DECAY` (như Khanh yêu cầu) — vấn đề đã tìm ra trước đó
(epsilon không kịp giảm về floor trong 4000 lifetime mặc định) **vẫn còn đó, chưa sửa** — chỉ
đổi cách resume giữa các lần chạy:
- File mới `src/rl/training_state.py`: `save_training_state()`/`load_training_state()` đọc/ghi
  `results/training_state.json` (epsilon hiện tại, tổng số lifetime đã train cộng dồn qua nhiều
  lần chạy, best_metric đã thấy — xem mục checkpointing bên dưới). Cố tình KHÔNG nhét vào
  `model_io.py` (file đó chỉ biết `net.state_dict()`, xem docstring — trộn trạng thái trời luyện
  vào đó sẽ lẫn "brain" với "một lần chạy cụ thể").
- `run_experiment.py`: nếu `training_state.json` tồn tại → dùng epsilon đã lưu để tiếp tục
  (không còn reset về 0.3 mỗi lần gọi nữa). Nếu chưa có file này nhưng đã có weight (chạy từ
  trước khi có tính năng này) → fallback 0.3 như cũ.
- **Thêm cờ `--reset-epsilon`**: giữ nguyên model đã train, chỉ reset epsilon về `EPSILON_START`
  — dùng khi muốn cho agent explore lại mà không mất weight. **Muốn train lại hoàn toàn từ đầu
  (cả model) thì chỉ cần xóa các file trong `results/`** — không cần thêm cờ riêng cho việc đó.
- `lifetimes_trained`/`best_metric` vẫn cộng dồn qua cả khi dùng `--reset-epsilon` (chỉ epsilon bị
  reset, không phải toàn bộ "lịch sử train").

**Chưa sửa (cố tình)**: tốc độ decay `EPSILON_DECAY=0.9995` vẫn không đủ nhanh để về đúng
`EPSILON_MIN=0.05` trong 4000 lifetime (cần ~5990 lifetime mới về đúng floor — tính trong hội thoại
 review trước). Với persist thì giờ điều này **không còn là vấn đề cấp bách** — nếu Khanh chia
 nhỏ training thành nhiều lần chạy cộng dồn đủ 5990+ lifetime tổng, epsilon vẫn sẽ về đúng
 floor vì không còn bị reset giữa chừng nữa.

**(c) — Reward shaping thưa do giới hạn theo tầm nhìn: KHÔNG sửa.** Đây không phải bug — là
 hệ quả đúng của việc bỏ rò rỉ thông tin toàn tri (xem mục trước). Sửa nó theo hướng "tăng
 `SHAPE_WEIGHT_*` để bù" là đoán mò không có dữ liệu; đúng hướng hơn là gắn với world/vision
 thật (mở rộng tầm nhìn) thì lại đụng kiến trúc (invalidate weight đang train). Để nguyên,
 chờ baseline số liệu thật rồi quyết định tiếp.

**(d) — Đã thêm `STARVATION_PENALTY` đối xứng với `HAZARD_PENALTY`** (`config.py`, mặc định cùng
 giá trị 1.00, cần tune riêng sau khi có dữ liệu). `Agent.apply_energy_cost()` giờ trả về `True`
 khi starvation damage xảy ra tick đó; `world_tick.TickResult` có thêm field `starved`;
 `compute_reward()` trừ `STARVATION_PENALTY` khi `starved=True`, y hệt cách hazard đã làm.
 **Lưu ý hệ quả**: với penalty này, `total_reward` trung bình mỗi lifetime sẽ THẤP HƠN hẳn so với
 các số liệu cũ đã ghi trong file này (verify thực tế: một lifetime ngắn với starvation kéo
 dài giờ cho reward âm nặng hơn nhiều so với trước đây) — đừng hoảng khi thấy `total_reward`
 âm to hơn hẳn so với các con số cũ, đây là thay đổi thang đo dự kiến, không phải regression.

**Thêm mới (không thuộc a-d, theo yêu cầu riêng của Khanh): best-model checkpointing.**
- `config.py`: `SAVE_EVERY=200` (lưu "latest" định kỳ, không chời đến hết run mới lưu — tránh
  mất toàn bộ tiến độ nếu crash giữa chừng) và `BEST_METRIC_WINDOW=50` (số lifetime trung bình
  trượt để xét "có phải best mới không").
- `run_experiment.py`: mỗi lifetime, trượt rolling mean `total_reward` qua `BEST_METRIC_WINDOW`
  lifetime gần nhất; nếu vượt best từng thấy → lưu riêng vào
  `results/trained_brain_conv_lstm_best.pt` (tách biệt với `trained_brain_conv_lstm.pt` “latest”).
  Mục đích đúng như Khanh nói: tránh overfit vào vài lifetime cuối cùng ngẫu nhiên/xấu — “latest”
  luôn phản ánh đúng hiện trạng training, “best” chỉ nhúc tới khi cả một cửa sổ lifetime
  (không phải 1 lifetime may mắn) xác nhận cải thiện.
- best-checkpoint chỉ bắt đầu kích hoạt sau khi đủ `BEST_METRIC_WINDOW` lifetime (tránh 1
  lifetime đầu ngẫu nhiên được phong "best" giả).

**Đã verify toàn bộ bằng chạy thật trong sandbox**: train mới → resume (epsilon đúng tại tục
0.9925→0.9925→0.9900) → `--reset-epsilon` (epsilon về 1.0, `lifetimes_trained` vẫn cộng dồn) →
 patch `SAVE_EVERY=8`/`BEST_METRIC_WINDOW=5` để xác nhận cả 2 cơ chế checkpoint thật sự kích
 hoạt đúng lúc → load lại `_best.pt` bằng `model_io` để chắc file tương thích. Phát hiện và sửa
 luôn 1 lỗi nhỏ trong lúc test: `training_state.json` lưu `best_metric=-inf` thành token
 `-Infinity` (Python đọc được nhưng không phải JSON chuẩn) — đổi sang lưu `null` + convert 2
 chiều.

---

## Đã thêm TERRAIN (2026-09) — soil/grass/wall/water

Theo yêu cầu "thế giới hợp lý hơn": World giờ có layer terrain **riêng biệt** với entity
(food/hazard) — cố định suốt 1 lifetime, không bị ăn/respawn như entity.

**Đã quyết định (theo Khanh chốt):**
- Water: đi được nhưng tốn phí gấp đôi (năng lượng + thời gian, xem bên dưới).
- Làm ngay, chấp nhận training hiện tại bị invalidate (xem lý do ở dưới).

**Kiến trúc:**
- Bảng mã ô mở rộng từ 6 lên 8 class: `0=unknown, 1=wall, 2=water, 3=soil, 4=grass,
  5=food_low, 6=food_high, 7=hazard` (`config.NUM_CELL_CLASSES`). Ghép chung 1 bảng mã với entity
  (Phương án A trong hội thoại thiết kế — không tách layer one-hot riêng) — mỗi ô hiển đúng
  1 code, entity ưu tiên hơn terrain nếu cùng ô.
- `World.terrain` (numpy array `[width, height]`) sinh 1 lần trong `_init_terrain()`
  (BEFORE `_init_entities()` — entity spawn giờ tránh ô wall): nền grass/soil random
  (`SOIL_FRACTION=0.30`), sau đó khắc `NUM_WATER_PATCHES=2` patch 2x2 và `NUM_WALL_SEGMENTS=2`
  đoạn thẳng dài `WALL_SEGMENT_LEN=3`. Ô trung tâm (spawn của agent) luôn được giữ sạch
  wall/water (`protected` set trong `_init_terrain`).
- **Không kiểm tra connectivity/reachability đầy đủ (BFS)** — chấp nhận rủi ro thấp vì
  world 10x10 và tổng diện tích wall/water cụt (tối đa ~6 wall + 8 water / 100 ô). Nếu sau
  này thấy agent hay bị kẹt, đây là chỗ cần quay lại.
- `World.step()`: chuyển `new_pos` dự định sang **không áp dụng** nếu ô đích là wall (agent
  "đụng tường", đứng nguyên chỗ nhưng vẫn tốn energy như mọi hành động). Hàm giờ trả về
  **4 giá trị**: `(new_pos, new_facing, event, entered_water)` — thay đổi chữ ký, chỉ có
  1 caller duy nhất (`world_tick.py`, đã xác nhận qua grep) nên an toàn.
- **Water cost**: `Agent.commit_action(action, water=...)` nhân đôi `ACTION_DURATION` cho
  forward/backward khi đích là water (`config.WATER_DURATION_MULTIPLIER=2`). Vì mỗi tick đều
  tốn `ENERGY_COST` như nhau bất kể action, kéo dài duration tự động kéo dài cả tổng năng
  lượng tiêu hao — 1 cơ chế duy nhất đáp ứng cả "tốn thời gian" lẫn "tốn năng lượng" như
  Khanh yêu cầu, không cần thêm cơ chế riêng.
- `world_tick.py`: **đổi thứ tự** `commit_action`/`cooldown_tick` ra SAU `world.step()` (trước đây
  ở TRƯỚC) để `commit_action` biết được `entered_water`. Đã xác nhận 2 hàm đó không đọc
  `agent.position` nên đổi thứ tự không ảnh hưởng gì khác.

**Visualization**: `live_viewer.py` vẽ nền terrain trước grid lines/entities (màu: wall xám,
water xanh dương, soil nâu, grass xanh lá); `visualize.py` (GIF matplotlib) vẽ terrain 1 lần
bằng `imshow` trước khi animate (terrain tĩnh cả lifetime, khỏi vẽ lại mỗi frame) —
`run_experiment.py` truyền `terrain=world.terrain` (lấy trực tiếp từ object `world` của lifetime
được ghi hình, không cần đổi chữ ký `run_episode()`). Đã verify trực quan bằng cách xuất 1
frame từ GIF test — màu đúng, dễ nhận biết.

**Hệ quả quan trọng — training hiện tại (nếu đang chạy) BỊ INVALIDATE**: `NUM_CELL_CLASSES`
đổi 6→8 đổi kích thước input của `ConvLSTMDQN` → `model_io.load_lstm_weights` sẽ tự động
phát hiện shape không khớp và raise `ValueError` → `run_experiment.py` tự catch và khởi tạo
brain mới thay vì crash (đã verify bằng checkpoint giả mô phỏng shape cũ). Lần chạy tiếp
theo của Khanh sẽ tự động bắt đầu lại từ brain mới — đúng như đã xác nhận trước khi làm.

**Đã verify toàn bộ bằng chạy thật trong sandbox**: sinh terrain + kiểm tra không entity nào
rơi vào wall → test trực tiếp `world.step()` cho cả wall-block và water-flag → test
`Agent.commit_action` nhân đôi duration đúng → full `run_experiment.py --lifetimes 15` chạy
sạch → `live_viewer.py` headless 500 tick với training bật, terrain render không lỗi.

**Chưa làm (ngoài phạm vi được yêu cầu "vài cái đơn giản")**:
- Line-of-sight/occlusion (wall không chặn tầm nhìn, agent vẫn "x-ray" xuyên tường thấy gì ở
  đáng sau)
- Soil/grass hiện tại **không khác nhau về cơ chế** (chỉ khác mã/màu) — chỗ để dành cho
  spawn-bias sau này nếu muốn
- Terrain không ảnh hưởng reward shaping trực tiếp (chỉ ảnh hưởng qua việc thay đổi
  số tick thực tế giữa các quyết định)

---

## Gom về 1 file model duy nhất: `results/best_model.pt` (2026-09)

Trước đó có 2 file `.pt` không nhất quán: `run_experiment.py` load/lưu vào
`trained_brain_conv_lstm.pt` ("latest"), `trained_brain_conv_lstm_best.pt` chỉ được GHI khi
có best mới (không ai đọc lại nó), và `live_viewer.py` lại load từ file "latest" riêng —
đúng như Khanh nhận ra, rườm rà và không nhất quán.

**Giờ chỉ còn 1 file duy nhất**: `results/best_model.pt`. Cả `run_experiment.py` và
`live_viewer.py` đều load từ đó, train tiếp trên đó, lưu lại vào đúng đó — không còn
"latest vs best" nữa.

**Đánh đổi cần biết (Claude chủ động nêu, không ẩn)**: cơ chế "best-model checkpointing
để tránh overfit" làm ở phiên trước (tách riêng file best khỏi latest, KHÔNG bao giờ bị ghi
đè bởi 1 snapshot tệ hơn) **không còn đúng nghia đó nữa** khi chỉ còn 1 file: `best_model.pt`
giờ vẫn có thể bị ghi đè bởi checkpoint định kỳ (`SAVE_EVERY`) kể cả khi checkpoint đó
 thực ra tệ hơn điểm tốt nhất đã thấy trước đó — file giờ chỉ còn bảo vệ được "mất tiến độ
 nếu crash giữa chừng", không còn bảo vệ được "train lâu quá bị tệ đi". `best_metric` vẫn
 được track/log trong `training_state.json` để tham khảo (in ra console khi có best mới), nhưng
 không còn gắn với việc bảo vệ file riêng nữa. Nếu sau này thấy cần lại sự bảo vệ đó, có
 thể thêm lại 1 file thứ 2 — nhưng giờ đơn giản hơn theo đúng yêu cầu.

**`run_experiment.py`**: lưu vào `best_model.pt` ở 2 thời điểm — định kỳ mỗi `SAVE_EVERY`
 lifetime (crash safety), VÀ ngay khi phát hiện rolling-mean reward vượt best cũ (không cần
 đợi đến mốc `SAVE_EVERY` tiếp theo mới lưu cải thiện thật).

**`live_viewer.py`**: đổi default `trained_weights_path` từ `trained_brain_conv_lstm.pt` sang
 `best_model.pt` — chỉ 1 dòng, logic save (phím S / auto-save khi thoát) không đổi, vẫn ghi
 đúng vào `self.weights_path`.

**Đã verify toàn bộ vòng đời đầy đủ trong sandbox**: `run_experiment.py --lifetimes 15`
 (brain mới) → chỉ sinh đúng 1 file `best_model.pt` → `live_viewer.py` load chính xác file đó
 (assert path), chạy 50 tick, lưu lại → `run_experiment.py --lifetimes 3` chạy tiếp, load
 lại đúng epsilon (0.9925→0.9910) và `lifetimes_trained` cộng dồn đúng (15→18) — xác nhận
 vòng khép kín hoạt động như Khanh muốn.

**Không còn file `.pt` cũ trên máy Khanh** tại thời điểm này (đã kiểm tra `results/` thật —
 chưa có `.pt` nào, vì tiến trình training đang chạy của Khanh dùng code cũ trong RAM chưa
 từng lưu) — không cần dọn/đổi tên gì thêm.

---

## Food phân bố rộng hơn + food_starter không tái tạo (2026-09)

**Mục đích**: Khanh nhận thấy agent hay "đi loạnh quanh không làm được gì" — nghi ngờ food quá
 thưa nên mới sinh ra khó nhìn thấy ngay. Muốn: (1) food trải đều hơn theo block 3x3, mỗi
 block ít nhất 1 food; (2) thêm 1 loại food đặc biệt, ăn xong mất luôn (không tái tạo), luôn
 đặt ngay trước mặt agent lúc sinh ra — để lifetime nào cũng chắc chắn ăn được ngay từ đầu.

**Kết quả mong đợi**: agent bớt cảm giác "không biết làm gì" ở đầu lifetime — luôn có ít
 nhất 1 thành công ngay từ tick đầu (ăn starter food), và dù đi hướng nào cũng nhanh chóng gặp
 food khác (không phải đi xuyên cả world mới thấy cái gì đó).

**Đã phát hiện khi đọc lại code**: World đã có sẵn hệ thống zone 3x3 (`ZONE_SIZE=3`) từ
 trước, nhưng chỉ round-robin **riêng từng loại food** (`food_low` và `food_high` riêng) qua các
 zone — với world 10x10/ZONE_SIZE=3 có **9 zone** nhưng chỉ 2+2=4 food tổng, nên "mỗi zone ≥ 1
 food" chưa bao giờ được đảm bảo thật (tối đa chỉ 4/9 zone có food). Đây là nguyên nhân gốc
 rễ đúng như Khanh nghi ngờ.

**Đã sửa**:
1. **Gộp food_low+food_high thành 1 pool round-robin chung** qua các zone (thay vì round-robin
   riêng từng loại) — đây mới là điều thực sự đảm bảo ≥ 1 food/zone khi tổng số food ≥ số
   zone.
2. **Tăng `NUM_FOOD_LOW` 2→5, `NUM_FOOD_HIGH` 2→4** (tổng 9 = đúng bằng số zone) — giờ đủ
   để mỗi zone có đúng ít nhất 1 food. **Lưu ý**: nếu sau này đổi `WORLD_SIZE` hoặc
   `World.ZONE_SIZE`, phải tính lại số zone và tăng food tương ứng (đã ghi comment rõ trong
   `config.py`) — không tự động giữ invariant này.
3. **Thêm entity mới `food_starter`**: `ENTITY_SPECS["food_starter"] = {"code": 8, "category":
   "food", "respawns": False}` — flag `respawns` mới (mặc định `True` qua `.get("respawns", True)`
   cho các loại cũ) để `World.step()` biết nên respawn hay `self.entities.remove(entity)` khi
   ăn xong.
4. **Vị trí cố định, không random**: thêm `World.spawn_point()` trả về `((width//2, height//2), 0)`
   — nguồn chân lý duy nhất cho "agent sinh ra ở đâu, quay mặt hướng nào", dùng chung bởi
   cả `Agent.__init__` (trước đây tự hardcode riêng `width//2, height//2` + `facing=0` — giờ
   gọi `world.spawn_point()`, tránh 2 nơi định nghĩa cùng 1 thứ rồi lệch nhau sau này) và
   `World._init_terrain()`/`_init_entities()`. `food_starter` đặt đúng `spawn_point() + facing
   vector` (ô ngay phía trước). Terrain generation giờ cũng bảo vệ luôn ô này (không phải
   chỉ ô trung tâm) khỏi wall/water, để vị trí đặt food_starter luôn hợp lệ.
5. `NUM_CELL_CLASSES` 8→9 (thêm channel cho code 8 = food_starter) — **lại invalidate brain
   đang có** (giống hệt lần thêm terrain) — chấp nhận, cùng tinh thần đang lải rải thiết kế
   world trước khi đầu tư train thật.
6. `FOOD_EFFECTS`/`FOOD_BONUS` cho `food_starter`: giống hệt `food_low` — không cố tình cho bổ
   dưỡng hơn, giá trị đặc biệt của nó nằm ở **khi nào/ở đâu** xuất hiện, không phải ăn bổ
   hơn.

**Visualization**: `visualize.py` (GIF) và `live_viewer.py` đều thêm marker riêng cho
`food_starter` (kim cương hồng/teal, khác hẳn food_low/high) + legend.

**Đã verify bằng chạy thật trong sandbox**: 20 seed khác nhau đều xác nhận mọi zone có
 ≥ 1 food và `food_starter` luôn đúng ô ngay trước mặt agent lúc spawn → test trực tiếp ăn
 `food_starter` xác nhận không respawn (biến mất hẳn, không trigger lại) → full
 `run_experiment.py --lifetimes 15` chạy sạch, số food ăn được ngay lt=0 (trước khi train gì)
 đã là 4 — cao hơn hẳn so với trước đây — → `live_viewer.py` headless không lỗi → xuất 1
 frame GIF kiểm tra trực quan: food dày đặc khắp world, food_starter đúng ô trước mặt
 spawn.

---

## Sửa tiếp: food_starter phải DÙNG CHUNG code với food_low, không được code riêng (2026-09)

**Mục đích**: Khanh chỉ ra đúng một điểm Claude làm sai tinh thần lúc đầu — khi thiết kế
 `food_starter` ở lượt trước, Claude cho nó 1 code quan sát riêng (8), tách biệt hoàn toàn với
 `food_low` (5) trong mắt network. Về mặt game-logic thì đúng (reward/dinh dưỡng đã giống
 `food_low` từ trước), nhưng **về mặt nhận thức của agent thì sai mục đích**: agent chỉ gặp
 `food_starter` đúng 1 lần duy nhất/lifetime — không đủ dữ liệu để network học riêng ý nghĩa
 của ký hiệu 8, nên "thành công đảm bảo ngay khi sinh" gần như vô tác dụng với việc học —
 đúng như Khanh nhận ra: "agent ăn xong mà giá trị hiển thị không giống food bình thường thì
 chả có tác dụng gì".

**Kết quả mong đợi**: network nhận ra `food_starter` ngay lập tức là "thứ food quen thuộc
 đã biết" (vì cùng ký hiệu với `food_low` mà nó được thấy rất nhiều lần mỗi lifetime), thay
 vì phải học riêng từ con số 0 một ký hiệu chỉ xuất hiện 1 lần duy nhất.

**Đã sửa**:
- `World.ENTITY_SPECS["food_starter"]["code"]`: 8 → **5** (giống hệt `food_low`). Vẫn là
  entity type riêng biệt về mixin logic (vị trí cố định, `respawns: False`) — chỉ có
  **cái mà network nhìn thấy** là giống nhau.
- `NUM_CELL_CLASSES`: 9 → **quay về 8** (không cần channel riêng nữa) — như vậy lần sửa
  này **không** invalidate brain (nếu đã có brain 8-class từ trước khi thêm terrain, giờ
  quay lại đúng kích thước đó); nhưng brain 9-class đã từng được tạo ở bước trước (nếu
  có) thì giờ lại không khớp nữa — `model_io` sẽ tự phát hiện và fallback brain mới như
  mọi lần (đã verify bằng checkpoint giả 9-class).
- `live_viewer.py`: bỏ entry code=8 khỏi `CELL_COLORS` (không còn tồn tại). Marker riêng
  (kim cương teal) cho `food_starter` ở **world map chính** vẫn giữ nguyên — vì đó là vẽ
  theo `entity["type"]` (cho người xem), không phải theo code quan sát (cho agent). Panel
  "AGENT LOCAL VISION" (mini grid) giờ tự động hiển thị `food_starter` **y hệt** `food_low`
  — đúng với nhận thức thực tế của agent.

**Điểm quan trọng rút ra**: có 2 lớp "hiển thị" khác nhau hoàn toàn trong project này — (1)
 **quan sát của agent** (local-view code → one-hot → network input, quyết định hành vi học
 được gì) và (2) **hiển thị cho người xem** (màu/marker trong GIF, live_viewer, hoàn toàn
 không ảnh hưởng đến việc học). Hai loại này có thể khác nhau hoàn toàn mà không sao —
 nhưng phải luôn rõ đâu là đâu khi thiết kế entity mới trong tương lai.

**Đã verify bằng chạy thật**: xác nhận `food_starter` code=5 = `food_low` code, local view
 thực sự hiển code 5 ở ô trước mặt spawn → full `run_experiment.py --lifetimes 15` chạy
 sạch → `live_viewer.py` headless không lỗi → test checkpoint giả 9-class bị từ chối đúng
 cách (không crash, fallback brain mới).

---

## Đổi sang MULTI-HOT thật (không còn mutually-exclusive) cho observation (2026-09)

**Mục đích**: Khanh muốn encode kiểu binary multi-channel thật sự — nhiều layer có thể cùng
 bật cho 1 ô (VD ô vừa là water vừa có food_low → cả 2 bit đều bật), thay vì 1 ô chỉ được
 1 mã duy nhất như trước (entity đè lên terrain, che mất thông tin terrain bên dưới). Lý do
 Khanh nêu: sau này sẽ còn nhiều thứ chồng lên nhau hơn nữa, nên cần chuyển sang kiểu này
 để agent học tốt hơn (và để kiến trúc scale được khi thêm layer mới sau này, không phải
 thiết kế lại encoding mỗi lần).

**Kết quả mong đợi**: network nhận đủ thông tin về một ô (cả terrain LẪN entity nếu có),
 thay vì chỉ thấy được 1 trong 2 — về lý thuyết giúp phân biệt được ô “food đặt trên
 water” (tốn thêm thời gian/năng lượng để đến) với “food đặt trên grass” (đến bình
 thường) — trước đây 2 trường hợp này nhìn HOÀN TOÀN GIỐNG NHAU qua quan sát (chỉ thấy
 code food, không thấy được terrain bị food che), nên agent không thể học được sự khác
 biệt đó dù thực tế chi phí khác nhau.

**Đã sửa**:
- Thêm `World.get_local_view_layers(position, facing)` — trả về 2 grid RIÊNG BIỆT
  (`terrain_grid`, `entity_grid`, entity dùng sentinel `-1` = không có) thay vì 1 grid đã
  bị collapse ưu tiên entity như `get_local_view()` cũ. **`get_local_view()` cũ vẫn giữ
  nguyên** — dùng riêng cho hiển thị (mini panel "AGENT LOCAL VISION" trong `live_viewer.py`),
  không ảnh hưởng gì đến nhận thức của network.
- `lstm_q_network.py`: `_class_vector(grid)` → `_multi_hot_grid(terrain_grid, entity_grid)`.
  Mỗi ô: bit terrain LUÔN bật (đúng 1 trong wall/water/soil/grass/unknown — những cái này
  vẫn loại trừ lẫn nhau), VÀ độc lập, bit entity bật thêm nếu có (`entity_grid != -1`).
  Ví dụ đúng như Khanh nêu: `water + food_low` → `[0,0,1,0,0,1,0,0]` (đã verify trực tiếp,
  bit‐khớp từng phiên bản).
- `encode_observation()` đổi chữ ký: `encode_observation(terrain_grid, entity_grid,
  internal_state)` thay vì `encode_observation(grid, internal_state)`. Cập nhật 3 nơi gọi:
  `world_tick.py` (x_next), `run_episode.py` (x đầu lifetime), `live_viewer.py`
  (`_reset_episode`).

**Điểm quan trọng: KHÔNG invalidate brain hiện có** — `NUM_CELL_CLASSES` vẫn là 8, kích
 thước vector vẫn 195 y hệt trước, nên `model_io`'s shape-check sẽ KHÔNG phát hiện gì cả (tải
 bình thường, không lỗi). **Nhưng ngữ nghĩa input đã đổi thật** — một brain đã train với
 encoding cũ (mutually-exclusive) sẽ nhận input theo phân phối khác hẳn (nhiều pattern 2-bit
 nó chưa từng thấy trong lúc train) — **nên vẫn khuyến nghị train lại từ đầu** dù kỹ thuật
 không bắt buộc (không crash) như các lần đổi `NUM_CELL_CLASSES` trước đây.

**Đã verify bằng chạy thật**: unit test trực tiếp `_multi_hot_grid` với 4 trường hợp
 (grass đơn, water+food_low, unknown, wall đơn) — khớp chính xác từng bit kỳ vọng → full
 `run_experiment.py --lifetimes 15` chạy sạch → `live_viewer.py` headless 200 tick với
 training bật, không lỗi.

---

## Food gieo hạt + mọc theo thời gian, thay cho "dịch chuyển tức thời" (2026-09)

**Mục đích**: Khanh muốn bỏ cơ chế "ăn xong dịch chuyển food tức thời sang ô khác" (cảm giác
 "phép thuật", không hợp lý) — thay bằng food thật sự **mọc theo thời gian**, mỗi ô có tỷ lệ
 riêng, và phải **gieo hạt trước** rồi mới có food (không phải roll xong ra food ngay). Đây
 cũng chính là điểm khiến soil/grass **khác nhau về cơ chế thật** (không chỉ khác màu như
 trước), đúng hướng đã để ngỏ từ lúc thêm terrain.

**Kết quả mong đợi**: world cảm giác "sống" hơn — food không phải vector cố định luôn
 có sẵn ở đâu đó, mà xuất hiện/biến mất theo một nhịp điệu tự nhiên; xem GIF/live_viewer
 sẽ thấy được "mầm" (chấm xanh nhạt nhỏ) trước khi nó thành food thật — trực quan hóa
 đúng quá trình sinh trưởng thay vì food "nổ ra" đột ngột.

**Chốt thiết kế (theo trả lời của Khanh)**:
- Loại food mới mọc: tỷ lệ cố định 70% food_low / 30% food_high, không phụ thuộc terrain
  hay loại food hàng xóm.
- 2 giai đoạn: **gieo hạt** (xác suất mỗi tick) → **chín** (đếm ngược thời gian cố định,
  không phải xác suất) — gần giống logic trồng cây thật: gieo là may rủi, nhưng đã gieo
  thì chắc chắn lớn sau 1 khoảng thời gian cố định (ngẫu nhiên trong khoảng 15-30 tick).
- Soil mọc NỂN nhanh hơn (tự nhiên, không cần food hàng xóm), grass LAN TRUYỀN nhanh hơn
  (bùng khi có food kề bên) — đúng ngược với đề xuất ban đầu của Claude, theo ý Khanh chọn.
- Hạt giống **không hiển thị cho agent** (không thêm observation channel) — chỉ là quá
  trình nền của world, agent chỉ "biết" khi nó đã thành food thật. Có thể nâng cấp sau
  nếu muốn agent học hành vi "chờ gần hạt sắp chín" (đổi lại phải thêm channel).

**Đã sửa**:
- `ENTITY_SPECS["food_low"/"food_high"]` thêm `"respawns": False` (giống `food_starter`) —
  **không còn dịch chuyển tức thời nữa**, ăn xong biến mất hẳn, food mới chỉ đến qua growth.
  `_respawn_entity`'s nhánh food zone-balance giờ thành dead code trong thực tế (chỉ còn
  hazard dùng) — để nguyên, giữ tổng quát cho tương lai, chỉ cập nhật comment.
- `World.seeds`: dict mới `{(x,y): ticks_remaining}`, không phải entity (không có code/category).
- `World._advance_food_growth()`: gọi 1 lần cuối mỗi `World.step()`, bất kể action gì (background
  process). Bước 1: đếm ngược + chín hạt đủ giờ, roll loại 70/30. Bước 2: nếu chưa chạm
  `MAX_FOOD_ON_WORLD=20`, tính xác suất gieo hạt **vector hoá bằng numpy cho cả grid mỗi tick**
  (không loop Python từng ô) — base rate theo terrain + bonus nếu ô kề (4 hướng) có food
  TRƯỞNG THÀNH (không tính hạt khác — chỉ food thật mới "lan truyền" được).
- Config mới: `FOOD_SEED_BASE_RATE`, `FOOD_SEED_SPREAD_BONUS` (dict `{"soil":, "grass":}`),
  `FOOD_SEED_MATURATION_MIN/MAX` (15/30), `FOOD_GROWTH_TYPE_SPLIT` (0.7/0.3), `MAX_FOOD_ON_WORLD=20`
  (trần "carrying capacity" — chỉ chặn GIEO HẠT MỚI, hạt đang chờ sẵn vẫn chín bình thường
  nên tổng food có thể nhính nhẹ trên mức trần ngay sau khi vài hạt chín cùng lúc).
- `visualize.py` + `live_viewer.py`: thêm marker riêng cho seed (chấm xanh vàng nhạt, nhỏ) —
  chỉ để người xem thấy, không ảnh hưởng agent.

**Đã verify bằng chạy thật**: xác nhận 2 giai đoạn đúng thứ tự (hạt trước, food sau, có
 độ trễ) → test thống kê tỷ lệ loại food (87 mẫu ra 81.6% thấy lệch, verify lại với
 100k mẫu ra đúng 70.05% — xác nhận không phải bug, chỉ nhiễu thống kê mẫu nhỏ) → test
 cap chặn gieo hạt mới đúng → chạy 2000 tick liên tục qua nhiều lifetime, food dao động
 ổn định quanh mức trần (18-21), KHÔNG tăng vô hạn → full `run_experiment.py --lifetimes 15`
 chạy sạch → xuất frame GIF thực tế thấy rõ chấm hạt giống đang chờ chín (tick 30) trước
 khi thành food.

**Không invalidate brain** — không đụng đến `NUM_CELL_CLASSES`/kích thước observation, chỉ
 đổi logic world-side. Nếu Khanh đã có brain từ bước multi-hot trước thì vẫn dùng được —
 nhưng hành vi "ăn xong food không dịch chuyển tức thời nữa" là thay đổi rất lớn về động
 lực world, nên kết quả học được của brain cũ (nếu có) có thể không còn phản ánh đúng.

---

## Giảm food + thêm reward theo age (2026-09)

**(1) Giảm food — Mục đích**: Khanh thấy world sau khi thêm growth cảm giác nhiều food quá.
**Sửa tối thiểu** (đúng yêu cầu "đừng thay đổi nhiều"): chỉ hạ 3 con số trong `config.py`,
 không động gì khác — `FOOD_SEED_BASE_RATE`/`FOOD_SEED_SPREAD_BONUS` giảm khoảng một nửa,
 `MAX_FOOD_ON_WORLD` 20→14. **Kết quả mong đợi**: food dao động thấp hơn hẳn. **Đã verify**:
 chạy 1800 tick liên tục, food giờ ổn định quanh 13-15 (trước đó ~18-21).

**(2) Reward theo age — trả lời câu hỏi của Khanh trước khi sửa**: đúng là trước đó **chết
 không có reward/penalty riêng nào** — tick chết tính giống hệt tick thường. Tín hiệu duy
 nhất liên quan đến "sống lâu" là `SURVIVAL_BONUS=0.001` cộng dồn mỗi tick — cả đời
 (150-400 tick) chỉ cộng dồn được ~0.15-0.4, không đáng kể so với 1 lần ăn food (3-5).

**Đã sửa (theo 2 câu trả lời của Khanh: thưởng 1 cục lúc chết, tỷ lệ thuận với age; và
 sống đủ hết `max_age` thì được thưởng thêm)**:
- `AGE_BONUS_PER_TICK=0.05`: cộng **1 lần duy nhất** lúc kết thúc (`done=True`), bằng
  `0.05 * age` — không phải cộng dồn mỗi tick như `SURVIVAL_BONUS`. Sống ~200 tick → +10
  (~2 lần ăn food), sống hết 400 tick → +20. Áp dụng **bất kể lý do chết** (hazard, đói,
  hay hết tuổi thọ).
- `MAX_AGE_SURVIVAL_BONUS=10.0`: cộng THÊM (ngoài age bonus) **chỉ khi** kết thúc vì sống
  đủ hết `max_age` VÀ health lúc đó vẫn > 0 (phân biệt với "chết do hết máu đúng lúc tuổi
  thọ hết", trường hợp đó vẫn tính là chết, không được thưởng thêm).
- `compute_reward()` thêm 3 tham số `done`, `age`, `survived_full_life` — cập nhật cả 2 nơi
  gọi (`run_episode.py`, `live_viewer.py`), truyền `survived_full_life=(result.done and
  agent.health > 0)`.

**Đã verify bằng chạy thật**: test trực tiếp `compute_reward` với 4 tình huống (tick thường,
 chết sớm vì hazard, chết sớm vì đói, sống hết `max_age`) — khớp đúng từng con số kỳ
 vọng; xác nhận sống hết đời (reward=30) rõ ràng tốt hơn chết sớm (reward=1.5) → full
 `run_experiment.py --lifetimes 15` chạy sạch → `live_viewer.py` headless không lỗi.

**Lưu ý**: thay đổi này **không invalidate brain** (không đụng NUM_CELL_CLASSES/observation),
 nhưng làm thang đo tổng `total_reward` tăng đáng kể (mỗi lifetime cộng thêm hàng chục
 điểm từ age bonus) — đừng so trực tiếp `total_reward` trước/sau thay đổi này, không phản
 ánh đúng "agent giỏi hơn hay tệ hơn", chỉ là đổi thang đo.

---

## 🌱 Reproduction / Ecosystem (2026-09) — thay đổi lớn nhất từ đầu dự án

### Mục đích & kết quả mong đợi

Khanh muốn thêm sinh sản: agent đủ năng lượng thì sinh ra agent con, kế thừa gần như
 y hệt weight cha + biến dị ngẫu nhiên ("gen") — để dần hình thành **1 hệ sinh thái**
 thật sự, thay vì chỉ 1 agent đơn độc sống-chết lặp lại. Đây chính là ranh giới
 **multi-agent** đã cảnh báo từ rất sớm trong dự án (lúc đánh giá "world xong bao nhiêu %").

**Kết quả mong đợi**: nhiều agent sống đồng thời, sinh sản/chết tự nhiên qua nhiều thế
 hệ, quan sát được cả **học cá nhân** (backprop trong đời) lẫn **tiến hoá quần thể**
 (chọn lọc gen qua các thế hệ) cùng lúc.

### 3 quyết định cốt lõi (theo Khanh chốt)
1. **Hybrid nature+nurture**: agent con vẫn học backprop trong lúc sống (không chỉ weight
   cố định kế thừa), CỘNG thêm kế thừa+biến dị lúc sinh.
2. Sinh sản **tự động** khi đủ năng lượng (không thêm action mới → không đổi kích thước
   network → không invalidate brain cũ vì lý do này).
3. Trần dân số `MAX_POPULATION=12`.

### Kiến trúc — phát hiện bất ngờ: World/world_tick KHÔNG CẦN SửA GÌ
Cả `World` lẫn `world_tick()` từ trước đến giờ đều nhận agent/brain/state qua tham số
 rõ ràng, không lưu "agent duy nhất" kiểu global state nào cả — nên gọi chúng nhiều lần/tick
 cho nhiều agent khác nhau trên CÙNG 1 world là đúng đắn ngay từ đầu, không cần đổi gì.
Đây là kết quả tự nhiên của nguyên tắc "state đi qua tham số, không phải biến toàn cục"
 đã giữ xuyên suốt dự án. Chỉ thêm đúng 1 method mới: `World.random_adjacent_cell(pos)`
 (tìm ô trống cạnh vị trí cha để đặt agent con).

### File mới: `src/rl/population.py`
- `Individual`: gói đầy đủ 1 agent sống — body (`Agent`), brain riêng (`TorchQAgent` — net +
  optimizer + target net **riêng từng con**), replay buffer riêng, policy riêng, `x/h/c` riêng,
  và `total_reward` tích luỹ (fitness score của riêng nó).
- `spawn_founder()` / `mutate_weights()` / `reproduce()`: tạo mới, biến dị (Gaussian noise
  `MUTATION_STD=0.05` trên **bản copy** weight cha, cha không bị đụng), sinh con (trừ năng
  lượng cha, chỉ khi còn chỗ dưới `MAX_POPULATION` và có ô trống cạnh cha).
- `ecosystem_step(world, population, auto_reseed=True)`: 1 tick cho **cả quần thể** — xáo
  thứ tự xử lý mỗi tick (công bằng tranh chấp food), gọi `world_tick()` y hệt code đơn-agent
  cho từng agent, xử lý chết/sinh, `auto_reseed=False` để tắt tự động hồi sinh khi tuyệt
  chủng (dùng cho `live_viewer.py`).

### 🐛 Lỗi quan trọng phát hiện + sửa trong lúc test (không phải chuyện nhỏ)

**1. Agent con không bao giờ học được gì cả (im lặng, không crash)**: `LSTMReplayBuffer`
 cũ chỉ cho `sample_windows()` từ các episode **đã đóng**, nhưng mỗi Individual trong
 ecosystem chỉ sống đúng **1 lần** (1 episode, luôn đang mở đến lúc chết) — nên
 `MIN_EPISODES=10` không bao giờ đúng, `learn_windows` không bao giờ được gọi. Đã thêm
 `LSTMReplayBuffer.current_tail(window_n)` — học từ **đuôi của chính đời đang diễn ra**
 thay vì sample từ pool đời cũ (không hề tồn tại với 1 cá thể). Đã verify trực tiếp:
 weight thật sự thay đổi qua các tick.

**2. Ghi đĩa liên tục mỗi tick nếu check sai cách**: nếu check "best mới" trên cả quần
 thể sống MỖI TICK, bất kỳ con đầu bảng nào đang sống cũng tự động "phá kỷ lục" mỗi tick
 (vì `total_reward` của nó luôn tăng) → ghi file liên tục, rất lãng phí. Đã sửa:
 `check_and_save_best()` tách riêng — check `deaths` (sự kiện hiếm, điểm đã chốt) MỖI
 tick, check quần thể đang sống chỉ MỖI `SAVE_EVERY` tick. Giảm từ hàng chục lần ghi/1000
 tick xuống còn **5 lần**.

**3. `live_viewer.py` ghi đè `best_model.pt` bằng model TỆ HƠN (nghiêm trọng nhất)**:
 phiên bản đầu tiên khởi tạo `best_metric_so_far = -inf` mỗi lần mở viewer, không đọc
 `training_state.json` cũ — verify thực tế: mở viewer sau khi đã có best=+19.10, chạy một
 lúc, kết thúc đã ghi đè thành -37.45 (**mất trắng tiến độ training trước đó**). Đã sửa:
 `_load_founder_net()` giờ đọc `training_state.json` ngay từ đầu, `best_metric_so_far` kế
 thừa đúng giá trị đã lưu. Đã verify lại: sau khi sửa, chạy 500 tick không hề làm giảm
 `best_metric_so_far` xuống dưới baseline đã load.

### `run_experiment.py` — viết lại hoàn toàn, chuyển sang ecosystem
Thế chỗ hẳn vòng lặp đơn-agent tuần tự cũ. Giờ: nạp `best_model.pt` làm founder (không biến
 dị — chỉ con mới bị biến dị, founder là đường tiếp nối trực tiếp từ brain đã lưu), chạy
 `--ticks` tick (thay `--lifetimes`), mỗi tick check best qua `check_and_save_best`. "Best" giờ
 là **best individual từng thấy trong toàn bộ quần thể** (`total_reward` của riêng nó), không
 còn là rolling-mean qua nhiều lifetime như trước. Vẫn chỉ 1 file `best_model.pt` duy nhất.
Đã verify: chạy mới → sinh nhiều thế hệ (tới gen 10 trong 1000 tick) → resume đúng (nạp
 lại weight + epsilon + best_metric) → tốc độ ghi đĩa hợp lý (5 lần/1000 tick).

### `live_viewer.py` — viết lại hoàn toàn, trở thành "xem thế giới" thay vì "xem 1 lifetime"
- Chạy **vô hạn**, không giới hạn lifetime — chỉ dừng khi **tuyệt chủng thật**
  (`auto_reseed=False`, khác với `run_experiment.py`) hoặc bấn **R** (nạp lại quần thể mới từ
  `best_model.pt`, không phải random trắng).
- Vẫn load/lưu **đúng cùng 1 file** `best_model.pt` với `run_experiment.py`.
- Vẽ **tất cả agent đang sống** trên world (không chỉ 1), mỗi con đúng hướng mặt — con
  đang được "spotlight" (agent già nhất còn sống, sticky theo id để không nhảy loạn) có
  viền vàng riêng để phân biệt.
- Panel bên phải: local vision + Q-value của spotlight, cộng thêm thống kê quần thể
  (số lượng, thế hệ cao nhất, sinh/chết trong phiên, best fitness đã lưu).
- Phím S / thoát: gọi `check_and_save_best` trên quần thể hiện tại.

**Đã verify đầy đủ bằng chạy thật (headless)**: reset sau tuyệt chủng hoạt động đúng, vẫn
 giữ best_metric qua reset, `draw()` không lỗi với nhiều agent cùng lúc.

### Phạm vi chưa làm (cố tình, theo đúng scope đã nói trước khi làm)
- Agent không nhìn thấy nhau, không va chạm/chặn ô của nhau — tương tác duy nhất là gián
  tiếp qua tranh food/không gian chung.
- `run_ecosystem.py` (phase6, xây trước `run_experiment.py` được viết lại) giờ **dư
  thừa** phần lớn chức năng (không có checkpointing) — vẫn để nguyên, chưa xóa vì không
  được yêu cầu.
- Không có cross-run persistence cho riêng quần thể (mỗi lần chạy `run_experiment.py`
  founder lại từ 1 brain, không lưu lại toàn bộ cây phả hệ).
