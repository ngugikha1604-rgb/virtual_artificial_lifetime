# Virtual Lifetime

> Một thế giới nhỏ có các sinh vật sống trong đó. Mở lên để xem chúng đi lại, tìm thức ăn, tránh nguy hiểm, lớn lên, sinh sản và thay đổi theo thời gian.

## Ý tưởng cốt lõi

Virtual Lifetime trước hết là một **thế giới thu nhỏ có thể quan sát**. Người dùng không mở chương trình để huấn luyện một mô hình hay theo dõi loss. Họ mở nó như mở cửa sổ nhìn vào một nơi đang sống: môi trường có nhịp điệu riêng, mỗi cá thể có trạng thái và lịch sử, và những gì xảy ra tiếp theo không được viết sẵn thành một màn trình diễn.

Hãy hình dung một terrarium kỹ thuật số: thức ăn xuất hiện và biến mất theo điều kiện của thế giới; địa hình ảnh hưởng đến việc đi lại; các sinh vật cần duy trì cơ thể, khám phá, phản ứng với những gì chúng cảm nhận được, già đi và có thể sinh con. Người dùng có thể quan sát cả thế giới hoặc theo dõi một cá thể, tua/chỉnh tốc độ, tạm dừng, và về sau tác động nhẹ lên môi trường nếu điều đó làm trải nghiệm thú vị hơn.

Mục tiêu là tạo ra những khoảnh khắc khiến người xem muốn nán lại: một cá thể tìm ra nơi có thức ăn, tránh một vùng nguy hiểm, quay lại một nơi quen thuộc, sinh con, hoặc biến mất khỏi thế giới. Hành vi không cần giống sinh học thật hay đạt một điểm số cao. Thế giới cần nhất quán, dễ đọc và tạo ra những diễn biến thú vị để quan sát.

## Hướng dẫn quan trọng cho AI hỗ trợ dự án

Khi đề xuất hoặc thực hiện thay đổi, hãy lấy các nguyên tắc sau làm chuẩn:

1. **Đây là mô phỏng để giải trí và quan sát, không phải dự án nghiên cứu hay benchmark RL.** Không mặc định rằng mục tiêu là tăng reward, loss, fitness, tốc độ hội tụ hoặc độ phức tạp của mạng.
2. **Thế giới là sản phẩm chính.** Ưu tiên tính liên tục, nhịp sống, tính dễ hiểu khi quan sát và cảm giác các cá thể đang tồn tại trong cùng một nơi.
3. **Cá thể là cư dân, không phải mẫu huấn luyện dùng một lần.** Danh tính, tuổi đời, nhu cầu, quan hệ với nơi chốn và lịch sử nên có ý nghĩa trong mô phỏng. Cái chết có thể xảy ra, nhưng không nên là cách mặc định để xoay vòng dữ liệu huấn luyện.
4. **Mở viewer phải đưa người dùng vào thế giới.** Viewer là trải nghiệm chính; không nên yêu cầu chạy một quy trình pretrain trước chỉ để có thứ đáng xem. Một thế giới mới vẫn cần có hoạt động thú vị từ đầu.
5. **Training là chi tiết triển khai tùy chọn.** Có thể dùng neural network, luật đơn giản hoặc kết hợp chúng. Chỉ giữ cơ chế học nếu nó làm hành vi sống động hơn theo cách người xem nhận ra được.
6. **Tách trạng thái thế giới khỏi trạng thái mô hình.** Tiến trình chính cần là thế giới/cư dân được lưu và tiếp tục; checkpoint model nếu còn cần thì là tài sản phụ, có thể thay thế hoặc bỏ qua.
7. **Tránh biến mọi hành vi thành tối ưu hóa điểm số.** Reward nội bộ có thể hỗ trợ một bộ điều khiển, nhưng không phải thước đo thành công của sản phẩm.
8. **Giữ thay đổi dễ quan sát và giải thích.** Khi sửa hành vi hoặc luật thế giới, nói rõ người dùng sẽ thấy khác biệt gì trong viewer.

Nếu một đề xuất làm cho project giống pipeline huấn luyện hơn nhưng không cải thiện thế giới mà người dùng nhìn thấy, thì đề xuất đó không đi đúng hướng.

## Trạng thái hiện tại

Code hiện có một mô phỏng grid 2D, cá thể có energy/health/age, thức ăn phát triển và biến đổi theo thời gian, terrain, hazard, controller luật, Conv-LSTM/DQN tùy chọn, sinh sản, save/load world và viewer Pygame. Hai entry point người dùng cần nhớ là:

- `train_brain.py` train/resume brain và lưu checkpoint.
- `run_world.py` mở một world mới hoặc tiếp tục world đã lưu.
- Các module trong `src/` là phần triển khai phụ; `experiments/pretrain_single_agent.py` được launcher training dùng nội bộ.

Các file kết quả như brain checkpoint và world save nằm trong `results/`; chúng không được viewer ghi đè lẫn nhau.

## Roadmap refactor

### Giai đoạn 1 — Định nghĩa trải nghiệm và đổi cấu trúc điều khiển

- Dùng live viewer làm entry point chính: khởi động là thấy thế giới, không phải chọn số tick training.
- Đưa bước tiến của thế giới vào một `Simulation`/`WorldSession` rõ ràng, sở hữu world, cư dân, thời gian mô phỏng, sự kiện và trạng thái tạm dừng/tốc độ.
- Viewer chỉ đọc trạng thái để vẽ và gửi lệnh người dùng như pause, speed, focus, reset hoặc save; logic thế giới không phụ thuộc Pygame.
- Thay các thuật ngữ hướng benchmark như “best individual”, “fitness” và “training run” trong giao diện bằng khái niệm thế giới như cá thể, tuổi, nhu cầu, sự kiện và thời gian.

**Kết quả mong muốn:** có thể chạy một thế giới mới và quan sát nó mà không cần gọi pretraining hay nghĩ về checkpoint mô hình.

### Giai đoạn 2 — Lưu và tiếp tục một thế giới

- Thiết kế world save gồm seed/trạng thái RNG, thời gian, terrain, entities/seeds, cá thể và các thuộc tính cần để tiếp tục đúng từ lần chạy trước.
- Lưu danh tính ổn định của cá thể, tuổi, vị trí, hướng, nhu cầu, trạng thái hành vi và các sự kiện/lịch sử cần thiết.
- Tách `world_save` khỏi model weights; cho phép mở lại thế giới kể cả khi đổi hoặc bỏ brain.
- Thêm autosave an toàn và lựa chọn tạo thế giới mới/mở thế giới gần nhất trong viewer.

**Kết quả mong muốn:** đóng rồi mở lại vẫn gặp cùng một thế giới và các cư dân đang sống tiếp từ trạng thái đã lưu.

### Giai đoạn 3 — Làm cho vòng sống có thể đọc được

- Kiểm tra và điều chỉnh cân bằng nhu cầu, thức ăn, hazard, di chuyển và tuổi thọ để cá thể có thời gian thể hiện hành vi.
- Biến sự kiện thành thứ người xem dễ nhận ra: vừa ăn, đói, bị thương, sinh con, khám phá, nghỉ, chết.
- Cải thiện cách chọn/follow một cá thể, bảng thông tin gọn, lịch sử sự kiện và điều khiển tốc độ/tạm dừng.
- Đảm bảo thế giới có diễn biến khi bắt đầu mới, không phụ thuộc vào việc một model đã học sẵn.

**Kết quả mong muốn:** người xem hiểu được điều gì vừa xảy ra và có lý do để tiếp tục theo dõi.

### Giai đoạn 4 — Tách AI hành vi khỏi hệ thống thế giới

- Tạo interface nhỏ cho bộ điều khiển cá thể: nhận observation và trạng thái nội bộ, trả về action.
- Có baseline luật đơn giản để thế giới luôn chạy được và làm chuẩn so sánh hành vi.
- Đưa Conv-LSTM/DQN thành một controller có thể bật/tắt/thay thế; việc học online là lựa chọn riêng, không bị buộc vào mỗi tick của viewer.
- Tách lưu/training state khỏi save của thế giới; không tự ghi đè một brain tốt chỉ vì cá thể có cumulative reward cao.
- Nếu giữ học, đánh giá bằng hành vi dễ quan sát và tùy chọn lưu phiên bản; reward/loss chỉ dùng để chẩn đoán nội bộ.

**Kết quả mong muốn:** thay đổi thuật toán không làm hỏng vòng sống, save/load hay viewer; thế giới không cần training để tồn tại.

### Giai đoạn 5 — Tăng chiều sâu thế giới theo từng lát nhỏ

- Thêm một cơ chế đời sống mỗi lần, ví dụ nhu cầu nghỉ ngơi, ghi nhớ nơi có thức ăn, vùng lãnh thổ, quan hệ gia đình hoặc mùa/chu kỳ môi trường.
- Mỗi cơ chế phải có biểu hiện quan sát được, trạng thái lưu được và quy tắc đủ đơn giản để giải thích.
- Chỉ mở rộng bản đồ/đồ họa sau khi vòng đời hiện tại tạo được các diễn biến đáng xem và chạy ổn định.

**Kết quả mong muốn:** thế giới có thêm câu chuyện emergent mà không biến thành một bộ sưu tập chỉ số khó hiểu.

## Cách chia code đích (khái niệm)

```text
world/          luật môi trường, thời gian, terrain, thức ăn và sự kiện
life/           cơ thể, nhu cầu, tuổi đời, sinh sản và danh tính cư dân
behavior/       interface controller, luật nền và controller học được
simulation/     sở hữu world + cư dân, tiến tick và quản lý save/load
viewer/         vẽ thế giới, chọn/follow cư dân và gửi thao tác người dùng
experiments/    công cụ tùy chọn để kiểm tra hoặc huấn luyện controller
```

Đây là hướng phân tách trách nhiệm, không phải yêu cầu đổi tên/thư mục ngay lập tức. Refactor nên diễn ra theo lát dọc có thể chạy được: mỗi giai đoạn giữ viewer hoạt động và cho thấy một cải thiện cụ thể.

## Tiêu chí thành công

- Tôi có thể mở viewer và thấy một thế giới đang hoạt động mà không cần chạy lệnh training trước.
- Tôi có thể quan sát hoặc theo dõi một cá thể và hiểu các nhu cầu/sự kiện chính của nó.
- Thế giới có thể được lưu, đóng, mở lại và tiếp tục.
- Cư dân có hành vi đủ đa dạng và thế giới tạo ra diễn biến thú vị ngay cả khi dùng controller luật đơn giản.
- Bất kỳ cơ chế học nào được giữ lại đều làm trải nghiệm quan sát tốt hơn; chỉ số training không phải mục tiêu sản phẩm.

## Chạy code hiện tại

Hai lệnh chính:

- `python train_brain.py --lifetimes 1000`: train/resume brain và lưu vào `results/best_model.pt`.
- `python run_world.py`: mở world để quan sát. Dùng `python run_world.py --load` để mở save gần nhất.

Các thư viện cần có gồm PyTorch, NumPy và Pygame.

Trong viewer, nhấn `S` để lưu thế giới vào `results/world_save.json` và `L` để mở lại save gần nhất. Save thế giới độc lập với `results/best_model.pt`; brain của từng resident (nếu có) nằm trong sidecar `world_save.json.brains.pt`.

Viewer cũng hiển thị resident đang được theo dõi, tuổi/thế hệ, hướng và hành động gần nhất, cùng lịch sử các sự kiện riêng của resident bên cạnh nhật ký chung của thế giới.

Các entry point ecosystem/pretrain cũ vẫn có thể dùng để khảo sát hệ thống hiện tại, nhưng roadmap hướng tới việc khiến chúng trở thành công cụ phụ trợ thay vì cách người dùng chính khởi động sản phẩm.

## Lịch sử kỹ thuật

Dự án từng bắt đầu như một bài tập tự cài đặt mạng neural bằng NumPy. Phần đó đã được thay bằng PyTorch. Lịch sử thay đổi chi tiết và các thí nghiệm trước đây nằm trong [`progress.md`](progress.md); hãy xem đó là nhật ký lịch sử kỹ thuật, còn README này mô tả mục tiêu hiện tại.
