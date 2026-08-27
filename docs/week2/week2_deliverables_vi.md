# Week 2 Deliverables - VAST-LoRA

NgÃ y cáº­p nháº­t: 2026-08-11  
Má»¥c tiÃªu: xÃ¢y ná»n low-rank algebra vÃ  simulator tÃ¡i láº­p Ä‘Æ°á»£c trÆ°á»›c khi triá»ƒn khai full VAST hoáº·c NVFlare.

## 1. Quyáº¿t Ä‘á»‹nh triá»ƒn khai

Week 2 váº«n giá»¯ nguyÃªn thesis core:

$$
\text{stale LoRA innovation}
\rightarrow
\text{intrinsic low-rank geometry}
\rightarrow
\text{current temporal subspace}
\rightarrow
\text{selective stale residual attenuation}.
$$

Thá»© tá»± triá»ƒn khai Ä‘Æ°á»£c giá»¯ nhÆ° sau:

```text
low-rank algebra kernel
-> pure PyTorch async simulator
-> diagnostic logging
-> Week 3-4 kill-test
-> VAST core
-> NVFlare integration
```

ChÆ°a triá»ƒn khai training LLM, chÆ°a tÃ­ch há»£p NVFlare, vÃ  chÆ°a thÃªm dynamic rank scheduler/fairness/privacy.

## 2. Low-rank algebra Ä‘Ã£ cÃ³

File chÃ­nh:

```text
src/riftlora/lowrank/core.py
```

CÃ¡c object/hÃ m chÃ­nh:

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

Server khÃ´ng nháº­n whole adapter nhÆ° update má»›i. Server pháº£i dá»±ng local innovation:

$$
D_i
=
G_{i,E}^{(v_i)}-G_{i,0}^{(v_i)}.
$$

Vá»›i LoRA factors:

$$
D_i
=
B_{i,E}A_{i,E}
-
B_{i,0}A_{i,0}.
$$

Code biá»ƒu diá»…n chÃ­nh xÃ¡c:

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

Do Ä‘Ã³:

$$
D_i=L_iR_i.
$$

### 2.2 Compact QR/SVD

ÄÆ°á»ng chÃ­nh khÃ´ng cáº§n dense reconstruction. Vá»›i:

$$
D_i=L_iR_i,
$$

ta QR:

$$
L_i=Q_{L,i}T_{L,i},
\qquad
R_i^\top=Q_{R,i}T_{R,i}.
$$

Ma tráº­n nhá»:

$$
M_i=T_{L,i}T_{R,i}^{\top}.
$$

SVD nhá»:

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

### 2.3 Projection vÃ  compatibility

Vá»›i reference bases $Q_L^t,Q_R^t$:

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

## 3. Async simulator Ä‘Ã£ cÃ³

File chÃ­nh:

```text
src/riftlora/asyncfl/simulator.py
```

Simulator hiá»‡n há»— trá»£:

- client profile: `client_id`, rank, sá»‘ máº«u, compute time, network time, jitter;
- dispatch version;
- arrival version;
- version staleness;
- event priority queue theo virtual finish time;
- buffer size;
- deterministic replay báº±ng seed.

Staleness Ä‘Æ°á»£c ghi theo:

$$
\tau_i=t-v_i.
$$

Trong Ä‘Ã³ $v_i$ lÃ  base version khi dispatch, cÃ²n $t$ lÃ  server version khi update vá» tá»›i server.

## 4. Partitioning vÃ  snapshot

ÄÃ£ cÃ³ deterministic data partitioning:

```text
src/riftlora/data/partitioning.py
```

Gá»“m:

- `iid_partition_indices`;
- `label_shard_partition_indices`.

ÄÃ£ cÃ³ snapshot store:

```text
src/riftlora/asyncfl/snapshots.py
```

Má»¥c tiÃªu cá»§a snapshot store lÃ  chuáº©n bá»‹ cho Week 3 khi má»—i client cáº§n train tá»« Ä‘Ãºng adapter version Ä‘Ã£ dispatch.

## 5. Script demo

Config:

```text
configs/week2_simulator.json
```

Cháº¡y:

```powershell
python scripts/run_week2_simulator.py
```

Output gá»“m:

- return order;
- dispatch versions;
- arrival versions;
- staleness values;
- staleness histogram;
- full records.

## 6. Tests Ä‘Ã£ cÃ³

```text
tests/lowrank/test_core.py
tests/test_async_simulator.py
tests/test_partitioning.py
```

CÃ¡c test chÃ­nh:

- innovation factorization khá»›p dense oracle;
- compact SVD reconstruct Ä‘Ãºng low-rank matrix;
- gauge-invariant dense result vÃ  singular spectrum;
- weighted sum khá»›p dense oracle;
- recompress theo rank budget;
- projection khá»›p dense oracle;
- full reference cho $\rho_i^t \approx 1$;
- simulator deterministic cÃ¹ng seed;
- buffered async chá»‰ tÄƒng version khi buffer Ä‘áº§y;
- partitioning deterministic vÃ  khÃ´ng máº¥t sample;
- snapshot store giá»¯ Ä‘Ãºng version.

Káº¿t quáº£ hiá»‡n táº¡i:

```text
12 passed
```

## 7. Handoff sang Week 3

Ná»n Week 2 Ä‘Ã£ Ä‘á»§ Ä‘á»ƒ bÆ°á»›c sang diagnostic data collection, nhÆ°ng cÃ²n cáº§n ná»‘i vá»›i LoRA training tháº­t.

Viá»‡c tiáº¿p theo:

1. Táº¡o minimal FedLoRA training loop nhá».
2. Gáº¯n `SnapshotStore` vá»›i adapter snapshots tháº­t.
3. Khi client return, dá»±ng:

$$
D_i=G_{i,E}^{(v_i)}-G_{i,0}^{(v_i)}.
$$

4. Log má»—i returned update:

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

5. ChÆ°a cáº§n full VAST transport trÆ°á»›c khi cÃ³ Ä‘á»§ stale innovations Ä‘á»ƒ kiá»ƒm tra $\rho_i^t$.


