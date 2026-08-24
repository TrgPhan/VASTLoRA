# Week 1 Deliverables - VAST-LoRA

Ngay cap nhat: 2026-08-10  
Trang thai: da tao skeleton folder va dong bang tam thoi literature map cho Week 1.

## 1. Muc tieu Week 1

Week 1 khong nen viet full algorithm ngay. Muc tieu dung hon la:

1. khoa scope nghien cuu;
2. sua lai cong thuc va ky hieu de co the code dung;
3. xac dinh novelty threat tu related work;
4. chon tech stack;
5. de xuat baseline MVP;
6. tao skeleton folder cho code va docs.

## 2. Folder da tao

```text
configs/
src/
src/vastlora/
src/vastlora/lowrank/
src/vastlora/lora/
src/vastlora/asyncfl/
src/vastlora/vast/
src/vastlora/baselines/
src/vastlora/eval/
src/vastlora/data/
src/vastlora/logging/
src/vastlora/nvflare_app/
tests/
tests/lowrank/
tests/vast/
scripts/
notebooks/
outputs/
docs/
docs/week1/
```

Tam thoi moi co `.gitkeep` de giu folder. Chua tao code module thuat toan, de tranh khoa sai design qua som.

## 3. Cong thuc can giu dung

### 3.1 LoRA layer

Voi layer $\ell$:

$$
W_0^{(\ell)}\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

LoRA adaptation:

$$
W^{(\ell)} = W_0^{(\ell)} + \Delta W^{(\ell)},
\qquad
\Delta W^{(\ell)} = s B^{(\ell)}A^{(\ell)}.
$$

Trong do:

$$
B^{(\ell)}\in\mathbb{R}^{d_{\text{out}}\times r},
\qquad
A^{(\ell)}\in\mathbb{R}^{r\times d_{\text{in}}},
\qquad
r \ll \min(d_{\text{out}},d_{\text{in}}).
$$

Neu client co rank khac nhau, nen control scale:

$$
\frac{\alpha_i}{r_i}=c
\qquad
\Rightarrow
\qquad
\alpha_i=c\,r_i.
$$

### 3.2 Staleness

Client $i$ duoc dispatch tai version $v_i$, server hien tai la version $t$:

$$
\tau_i=t-v_i.
$$

Freshness coefficient:

$$
\mu_i=e^{-\lambda\tau_i}.
$$

### 3.3 Local innovation, khong phai whole adapter

Adapter luc dispatch:

$$
G_{i,0}^{(v_i)}=B_{i,0}A_{i,0}.
$$

Adapter sau local training:

$$
G_{i,E}^{(v_i)}=B_{i,E}A_{i,E}.
$$

Message dung de gui ve server phai la innovation:

$$
D_i=G_{i,E}^{(v_i)}-G_{i,0}^{(v_i)}
$$

hay:

$$
D_i=B_{i,E}A_{i,E}-B_{i,0}A_{i,0}.
$$

Low-rank exact factorization:

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

Do do:

$$
D_i=L_iR_i,
\qquad
\operatorname{rank}(D_i)\le 2r_i.
$$

### 3.4 Compact SVD khong dense reconstruction

Thin QR:

$$
L_i=Q_{L,i}T_{L,i},
\qquad
R_i^\top=Q_{R,i}T_{R,i}.
$$

Shape dung:

$$
Q_{L,i}\in\mathbb{R}^{d_{\text{out}}\times k_{L,i}},
\qquad
Q_{R,i}\in\mathbb{R}^{d_{\text{in}}\times k_{R,i}},
\qquad
k_{L,i},k_{R,i}\le 2r_i.
$$

Ma tran nho:

$$
M_i=T_{L,i}T_{R,i}^{\top}
\in
\mathbb{R}^{k_{L,i}\times k_{R,i}}.
$$

Compact SVD:

$$
M_i=P_i\Sigma_iQ_i^\top,
$$

voi:

$$
P_i\in\mathbb{R}^{k_{L,i}\times m_i},
\qquad
Q_i\in\mathbb{R}^{k_{R,i}\times m_i},
\qquad
m_i\le \min(k_{L,i},k_{R,i}).
$$

Suy ra:

$$
D_i=U_i\Sigma_iV_i^\top,
\qquad
U_i=Q_{L,i}P_i,
\qquad
V_i=Q_{R,i}Q_i.
$$

### 3.5 Temporal reference subspace

History accepted innovations:

$$
\mathcal{H}_t
=
\{
\Delta G_{t-H+1},
\dots,
\Delta G_t
\}.
$$

Moi accepted increment:

$$
\Delta G_j=U_j\Sigma_jV_j^\top.
$$

Recency weight:

$$
\gamma_h=
\frac{e^{-\delta h}}
{\sum_{q=0}^{H-1}e^{-\delta q}},
\qquad
h=0,\dots,H-1.
$$

Build:

$$
M_L^t=
\left[
\sqrt{\gamma_0}U_t,
\sqrt{\gamma_1}U_{t-1},
\dots
\right],
$$

$$
M_R^t=
\left[
\sqrt{\gamma_0}V_t,
\sqrt{\gamma_1}V_{t-1},
\dots
\right].
$$

Top singular vectors:

$$
Q_L^t=\operatorname{TopSVD}_{R_L}(M_L^t),
\qquad
Q_R^t=\operatorname{TopSVD}_{R_R}(M_R^t).
$$

Projectors:

$$
P_L^t=Q_L^t(Q_L^t)^\top,
\qquad
P_R^t=Q_R^t(Q_R^t)^\top.
$$

### 3.6 Compatibility score

Projection:

$$
D_i^\parallel=P_L^tD_iP_R^t.
$$

Coordinate core:

$$
C_i^t=(Q_L^t)^\top U_i\Sigma_iV_i^\top Q_R^t.
$$

Do do:

$$
D_i^\parallel=Q_L^tC_i^t(Q_R^t)^\top.
$$

Energy:

$$
\|D_i^\parallel\|_F^2=\|C_i^t\|_F^2,
\qquad
\|D_i\|_F^2=\|\Sigma_i\|_F^2.
$$

VAST compatibility:

$$
\rho_i^t=
\frac{\|C_i^t\|_F^2}
{\|\Sigma_i\|_F^2+\epsilon},
\qquad
0\le\rho_i^t\le1.
$$

### 3.7 VAST transport

Residual:

$$
D_i^\perp=D_i-D_i^\parallel.
$$

Selective stale residual attenuation:

$$
\mathcal{T}_t(D_i)
=
D_i^\parallel+\mu_iD_i^\perp.
$$

Equivalently:

$$
\mathcal{T}_t(D_i)
=
\mu_iD_i+(1-\mu_i)D_i^\parallel.
$$

Retained energy, bo qua $\epsilon$ nho trong mau cua $\rho_i^t$:

$$
\frac{\|\mathcal{T}_t(D_i)\|_F^2}
{\|D_i\|_F^2}
=
\rho_i^t+\mu_i^2(1-\rho_i^t).
$$

## 4. Ket luan literature Week 1

Khong nen claim:

- VAST la phuong phap dau tien cho heterogeneous-rank FedLoRA.
- VAST la phuong phap dau tien align LoRA factors/subspaces.
- VAST la phuong phap dau tien giai gauge ambiguity.
- VAST la phuong phap dau tien asynchronous federated LLM fine-tuning.

Claim an toan hon:

> VAST-LoRA nghien cuu stale LoRA innovation trong asynchronous FedLoRA duoi rank heterogeneity. Diem chinh la correction rule data-free tren intrinsic low-rank geometry, giu phan compatible voi temporal reference subspace va chi attenuate stale residual.

## 5. Tech stack khuyen dung

### Core research

```text
Python 3.10 hoac 3.11
PyTorch
Transformers
PEFT
Accelerate
bitsandbytes
NumPy
SciPy
pandas hoac polars
scikit-learn
statsmodels
pytest
```

### Experiment management

```text
Hydra hoac OmegaConf
W&B hoac MLflow
matplotlib
seaborn
rich/tqdm
```

### Federated phase

```text
NVFlare 2.8.x
Docker
optional: Kubernetes
```

Khuyen nghi thu tu:

1. Week 1-6: pure PyTorch simulator.
2. Week 7: NVFlare local POC.
3. Sau khi algorithm on dinh: Docker/multi-process.
4. Kubernetes chi de demo system, khong dua vao novelty.

## 6. Model va dataset Week 1 decision

Model kill-test dau tien:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Training mode:

```text
QLoRA 4-bit NF4
bf16 compute
gradient checkpointing
sequence_length = 512
per_device_batch_size = 1
gradient_accumulation_steps = 4
target_modules = ["q_proj", "v_proj"]
```

Client ladder:

```text
Stage 0: N = 4, debug
Stage 1: N = 8, rank = 8, IID, controlled staleness
Stage 2: N = 8, ranks = {4, 8, 16, 32}
Stage 3: N = 12 hoac 16, non-IID
```

Dataset nen bat dau:

- diagnostic classification/short-form task: AG News, 20 Newsgroups, hoac GLUE subset;
- instruction tuning sau: Alpaca-style, Dolly-style, SuperNI-style.

## 7. Baseline MVP

Bat buoc:

1. Sync FedAvg-LoRA.
2. Naive Async-LoRA.
3. Freshness-only Async-LoRA:

$$
D_i^{\text{fresh}}=\mu_iD_i.
$$

4. Buffered Async-LoRA / FedBuff-style.
5. Heterogeneous-rank baseline.
6. GLoRA-like gauge-aware aggregation.
7. GLoRA-like + freshness-only.

De sau neu du thoi gian:

- AlignFed-compatible comparison;
- FedSteer-inspired projection baseline;
- FedRot-LoRA/FLoRG-inspired alignment baseline.

## 8. Week 2 handoff

Viec nen lam tiep:

1. Tao `pyproject.toml` va package skeleton neu bat dau code.
2. Implement event-driven async simulator.
3. Implement `LowRankMatrix` representation.
4. Implement exact innovation factorization.
5. Viet dense oracle tests cho low-rank algebra.
6. Chua implement full VAST truoc khi co simulator va logging.

## 9. Dieu kien GO/NO-GO cho Week 3-4

Tinh true post-hoc utility:

$$
u_i
=
\mathcal{L}(W_t)
-
\mathcal{L}(W_t+\eta D_i).
$$

Tinh transported utility:

$$
u_i^{\text{VAST}}
=
\mathcal{L}(W_t)
-
\mathcal{L}(W_t+\eta\mathcal{T}_t(D_i)).
$$

So sanh model:

$$
u_i\sim\tau_i
$$

voi:

$$
u_i\sim\tau_i+\rho_i.
$$

GO neu $\rho_i$ them signal sau khi control $\tau_i$ va VAST giam harmful-update rate trong stale regime.

NO-GO neu $\rho_i$ gan nhu khong co predictive value qua nhieu seed/task. Khi do pivot thanh empirical study ve gioi han cua parameter-space geometry cho stale federated adapters.

## 10. Tai lieu nguon

- GLoRA: https://arxiv.org/abs/2605.06733
- AlignFed: https://arxiv.org/abs/2606.08197
- FedSteer: https://arxiv.org/abs/2606.10124
- FedRot-LoRA: https://arxiv.org/abs/2602.23638
- FLoRG: https://arxiv.org/abs/2602.17095
- HetLoRA: https://aclanthology.org/2024.emnlp-main.717/
- FedEx-LoRA: https://aclanthology.org/2025.acl-long.67/
- FLoRA: https://proceedings.neurips.cc/paper_files/paper/2024/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html
- SDFLoRA: https://arxiv.org/abs/2601.11219
- PreLort: https://arxiv.org/pdf/2606.15963
- FSLoRA: https://arxiv.org/abs/2501.19389
- QLoRA: https://arxiv.org/abs/2305.14314

