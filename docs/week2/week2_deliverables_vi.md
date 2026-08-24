# Week 2 Deliverables - VAST-LoRA

Ngày cập nhật: 2026-08-11  
Mục tiêu: xây nền low-rank algebra và simulator tái lập được trước khi triển khai full VAST hoặc NVFlare.

## 1. Quyết định triển khai

Week 2 vẫn giữ nguyên thesis core:

$$
\text{stale LoRA innovation}
\rightarrow
\text{intrinsic low-rank geometry}
\rightarrow
\text{current temporal subspace}
\rightarrow
\text{selective stale residual attenuation}.
$$

Thứ tự triển khai được giữ như sau:

```text
low-rank algebra kernel
-> pure PyTorch async simulator
-> diagnostic logging
-> Week 3-4 kill-test
-> VAST core
-> NVFlare integration
```

Chưa triển khai training LLM, chưa tích hợp NVFlare, và chưa thêm dynamic rank scheduler/fairness/privacy.

## 2. Low-rank algebra đã có

File chính:

```text
src/vastlora/lowrank/core.py
```

Các object/hàm chính:

```text
LowRankMatrix
CompactSVD
exact_lora_innovation
compact_svd
weighted_sum
recompress
build_temporal_reference
project_to_reference
```

### 2.1 Exact innovation factorization

Server không nhận whole adapter như update mới. Server phải dựng local innovation:

$$
D_i
=
G_{i,E}^{(v_i)}-G_{i,0}^{(v_i)}.
$$

Với LoRA factors:

$$
D_i
=
B_{i,E}A_{i,E}
-
B_{i,0}A_{i,0}.
$$

Code biểu diễn chính xác:

$$
L_i=
\begin{bmatrix}
B_{i,E} & B_{i,0}
\end{bmatrix},
\qquad
R_i=
\begin{bmatrix}
A_{i,E}\\
-A_{i,0}
\end{bmatrix}.
$$

Do đó:

$$
D_i=L_iR_i.
$$

### 2.2 Compact QR/SVD

Đường chính không cần dense reconstruction. Với:

$$
D_i=L_iR_i,
$$

ta QR:

$$
L_i=Q_{L,i}T_{L,i},
\qquad
R_i^\top=Q_{R,i}T_{R,i}.
$$

Ma trận nhỏ:

$$
M_i=T_{L,i}T_{R,i}^{\top}.
$$

SVD nhỏ:

$$
M_i=P_i\Sigma_iQ_i^\top.
$$

Suy ra compact SVD:

$$
D_i=U_i\Sigma_iV_i^\top,
\qquad
U_i=Q_{L,i}P_i,
\qquad
V_i=Q_{R,i}Q_i.
$$

### 2.3 Projection và compatibility

Với reference bases $Q_L^t,Q_R^t$:

$$
D_i^\parallel=P_L^tD_iP_R^t.
$$

Coordinate core:

$$
C_i^t=(Q_L^t)^\top U_i\Sigma_iV_i^\top Q_R^t.
$$

Compatibility score:

$$
\rho_i^t
=
\frac{\|C_i^t\|_F^2}
{\|\Sigma_i\|_F^2+\epsilon}.
$$

## 3. Async simulator đã có

File chính:

```text
src/vastlora/asyncfl/simulator.py
```

Simulator hiện hỗ trợ:

- client profile: `client_id`, rank, số mẫu, compute time, network time, jitter;
- dispatch version;
- arrival version;
- version staleness;
- event priority queue theo virtual finish time;
- buffer size;
- deterministic replay bằng seed.

Staleness được ghi theo:

$$
\tau_i=t-v_i.
$$

Trong đó $v_i$ là base version khi dispatch, còn $t$ là server version khi update về tới server.

## 4. Partitioning và snapshot

Đã có deterministic data partitioning:

```text
src/vastlora/data/partitioning.py
```

Gồm:

- `iid_partition_indices`;
- `label_shard_partition_indices`.

Đã có snapshot store:

```text
src/vastlora/asyncfl/snapshots.py
```

Mục tiêu của snapshot store là chuẩn bị cho Week 3 khi mỗi client cần train từ đúng adapter version đã dispatch.

## 5. Script demo

Config:

```text
configs/week2_simulator.json
```

Chạy:

```powershell
python scripts/run_week2_simulator.py
```

Output gồm:

- return order;
- dispatch versions;
- arrival versions;
- staleness values;
- staleness histogram;
- full records.

## 6. Tests đã có

```text
tests/lowrank/test_core.py
tests/test_async_simulator.py
tests/test_partitioning.py
```

Các test chính:

- innovation factorization khớp dense oracle;
- compact SVD reconstruct đúng low-rank matrix;
- gauge-invariant dense result và singular spectrum;
- weighted sum khớp dense oracle;
- recompress theo rank budget;
- projection khớp dense oracle;
- full reference cho $\rho_i^t \approx 1$;
- simulator deterministic cùng seed;
- buffered async chỉ tăng version khi buffer đầy;
- partitioning deterministic và không mất sample;
- snapshot store giữ đúng version.

Kết quả hiện tại:

```text
12 passed
```

## 7. Handoff sang Week 3

Nền Week 2 đã đủ để bước sang diagnostic data collection, nhưng còn cần nối với LoRA training thật.

Việc tiếp theo:

1. Tạo minimal FedLoRA training loop nhỏ.
2. Gắn `SnapshotStore` với adapter snapshots thật.
3. Khi client return, dựng:

$$
D_i=G_{i,E}^{(v_i)}-G_{i,0}^{(v_i)}.
$$

4. Log mỗi returned update:

```text
client_id
base_version
arrival_version
tau
rank
num_samples
virtual_latency
update_fro_norm
effective_rank
rho_left
rho_right
rho_two_sided
raw_update_utility
```

5. Chưa cần full VAST transport trước khi có đủ stale innovations để kiểm tra $\rho_i^t$.

