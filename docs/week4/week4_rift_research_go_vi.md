# Week 4 - RIFT-LoRA research GO

Ngay chay: 2026-08-25

## Verdict

**GO de tiep tuc nghien cuu** cho mot claim hep:

> RIFT-LoRA giam rui ro cua late updates trong non-IID, high-staleness Async
> FedLoRA bang cach loc tung intrinsic singular component theo objective signal
> va chi chap nhan buoc van pass mot paired calibration-loss gate.

Day chua phai final thesis GO tren nhieu task/model, va khong phai claim RIFT co
accuracy cao hon moi FedLoRA method trong moi setting.

## RIFT la gi?

Ten lam viec: **Risk-filtered Intrinsic Federated Transport (RIFT-LoRA)**.

Voi exact stale innovation cua mot LoRA layer:

```text
D = B_final A_final - B_dispatch A_dispatch
  = sum_j sigma_j u_j v_j^T
```

RIFT dung calibration gradient `G` de cham tung component:

```text
gain_j = -sigma_j u_j^T G v_j
```

Chi component co `gain_j > 0` duoc giu. Server thu cac scale
`1, 1/2, 1/4, 1/8` tren mot calibration split thu hai, roi chon candidate co
paired mean loss change am nhat. Neu khong candidate nao giam calibration loss,
update bi reject.

Hai calibration split khong giao voi federated client pool va khong dung GLUE
validation labels:

- gradient split: 256 SST-2 train examples, indices 5000-5511;
- gate split: 256 examples con lai trong cung prefix;
- client pool bat dau tu index 5512;
- final evaluation dung toan bo 872 validation examples.

## Vi sao day la mot thesis khac voi cac doi thu?

- FedEx-LoRA sua exactness cua aggregation.
- FedRot-LoRA, FLoRG va GLoRA sua factor/subspace representation va alignment.
- FSLoRA va cac heterogeneous-rank methods sua resource/rank compatibility.
- SDFLoRA tach shared/private components cho personalization va privacy.
- AlignFed dung calibration set cho cross-version semantic alignment va fairness.
- RIFT khong co gang thay the cac lop tren. No la mot **late-update safety layer**
  co the boc ngoai mot intrinsic candidate update: loc component co predicted
  descent, scale, gate, hoac reject.

AdaLoRA la cam hung truc tiep cho viec xem singular directions nhu cac don vi co
importance khac nhau. Diem moi can bao ve cua RIFT khong phai "rank pruning",
ma la objective-aware rank filtering cua exact stale LoRA innovations ket hop
voi mot disjoint paired gate trong asynchronous FL.

## Six-seed held-out confirmation

Setting: BERT-tiny, SST-2, 10 clients, label-shard non-IID, client ranks
4/8/16, mean staleness cao, 20 measured returns sau 8 warmup returns.

| Method | Final accuracy | Final loss | Harmful updates | Harmful late updates (`tau >= 8`) | Acceptance |
|---|---:|---:|---:|---:|---:|
| RIFT | **73.777%** | **0.543222** | **4.17%** | **0.00%** | 50.83% |
| Freshness | 72.611% | 0.552068 | 50.00% | 44.44% | 100% |
| Exact / FedEx-style innovation | 72.496% | 0.553451 | 55.00% | 52.78% | 100% |
| VAST | 72.515% | 0.551310 | 50.83% | 44.44% | 100% |
| Projection / MTIP-style | 73.433% | 0.546267 | 56.67% | 52.78% | 100% |

Paired results:

- RIFT thang freshness accuracy 5/6 seed, mean `+1.166 pp`;
- RIFT thang freshness final loss 6/6 seed, mean gain `0.008846`;
- RIFT thang projection final loss 6/6 seed, mean gain `0.003045`;
- late harmful-update rate cua RIFT la 0% tren ca 6 seed;
- RIFT van tao positive loss progress tren 6/6 seed, nen khong phai nghiem
  `reject everything`;
- bo seed 113 co gain lon bat thuong, mean accuracy gain vs freshness van xap xi
  `+0.37 pp`.

Analyzer tai `scripts/analyze_week4_rift.py` pass tat ca frozen GO gates va ghi
artifact vao `outputs/week4_rift_analysis/`.

## Ablation tren ba development seeds

| Variant | Final accuracy | Final loss | Harmful updates | Harmful late updates | Acceptance |
|---|---:|---:|---:|---:|---:|
| Full RIFT | **73.662%** | 0.543654 | **0.00%** | **0.00%** | 60.00% |
| Gate-only, khong loc rank | 73.547% | 0.543890 | 10.00% | 16.67% | 45.00% |
| Filter-only, khong gate | **73.662%** | **0.543299** | 16.67% | 11.11% | 100.00% |

Filter-only co loss trung binh nhe hon trong development, nhung khong dat safety
claim. Full RIFT la diem trade-off hop ly: giu accuracy, loai late harm, va khong
reject tat ca update.

## QNLI cross-task confirmation

Khong doi RIFT rule hay hyperparameter, chi doi checkpoint/dataset sang
BERT-tiny/QNLI va dung ba seed 131/149/163.

| Method | Final accuracy | Final loss | Harmful updates | Harmful late updates | Acceptance |
|---|---:|---:|---:|---:|---:|
| RIFT | **74.121%** | **0.538466** | **1.67%** | **5.56%** | 98.33% |
| Freshness | 71.517% | 0.565783 | 48.33% | 44.44% | 100% |
| Exact / FedEx-style innovation | 70.475% | 0.573961 | 46.67% | 44.44% | 100% |
| VAST | 71.419% | 0.565118 | 46.67% | 44.44% | 100% |
| Projection / MTIP-style | 71.615% | 0.563886 | 45.00% | 38.89% | 100% |

RIFT thang accuracy va loss 3/3 seed truoc ca bon matched baselines. Mean gain:

- vs freshness: `+2.604 pp` accuracy, `+0.027317` loss;
- vs exact: `+3.646 pp` accuracy, `+0.035495` loss;
- vs VAST: `+2.702 pp` accuracy, `+0.026653` loss;
- vs projection: `+2.507 pp` accuracy, `+0.025420` loss.

Normal 95% CI cua accuracy gain vs ca bon baseline deu nam tren 0. QNLI chi co
ba seed, nen CI nay la supporting evidence, khong thay the them seeds.

## QNLI 60-event stress test

De kiem tra RIFT co chi giu checkpoint trong 20 events hay khong, cung ba seed
duoc chay 60 measured returns. Day la post-confirmation stress test, khong dung
de tune lai method.

| Method | Final accuracy | Final loss | Harmful updates | Harmful late updates | Acceptance |
|---|---:|---:|---:|---:|---:|
| RIFT | **74.674%** | **0.529083** | **8.89%** | **10.53%** | 73.33% |
| Freshness | 70.378% | 0.580396 | 52.78% | 50.88% | 100% |

RIFT thang accuracy va loss 3/3 seed, mean accuracy gain `+4.297 pp` va mean
loss gain `+0.051313`. Safety khong tuyet doi: late harm tang tu 5.56% o 20
events len 10.53% o 60 events. Vi vay thesis nen claim **risk reduction**, khong
claim "zero harmful updates" cho moi training horizon.

## Tai sao chi la research GO?

1. Hai task hien tai van cung mot BERT-tiny backbone; chua co generative/LLM task.
2. SST-2 95% normal CIs cua mean accuracy/loss gain van cat 0 do chi co 6 seed;
   QNLI signal manh hon nhung moi co 3 seed.
3. `z=0` la empirical paired-loss gate, chua phai probabilistic safety guarantee.
4. RIFT can 512 labeled server examples va them server forward/backward passes.
5. Chua co faithful matched implementation cua GLoRA, FLoRG, FSLoRA, SDFLoRA
   va AlignFed trong simulator nay.
6. Calibration shift co the lam gate sai; day phai la stress test bat buoc.

Vi vay cau noi dung la: **GO cho huong nghien cuu va do an**, chua GO cho paper
claim "vuot tat ca doi thu".

## Frozen thesis question de di tiep

> Can a calibration-assisted, gauge-invariant rank-wise safety layer reduce the
> harmful effect of late updates in heterogeneous-rank asynchronous FedLoRA
> while retaining enough updates to preserve optimization progress?

Primary metric khong con la accuracy don le. Thu tu metric:

1. late harmful-update rate va cumulative late harm;
2. worst-step loss increase;
3. acceptance rate va utility per accepted update;
4. loss-area-under-trajectory / time-to-target;
5. final loss, calibration va accuracy non-inferiority.

## Gate tiep theo

1. Port RIFT nhu wrapper tren FedRot/FedEx/GLoRA-like candidate updates.
2. Mo rong QNLI len 6-10 seed va chay mot task generative nho; giu
   calibration/client/eval disjoint.
3. Stress-test calibration shift, calibration size 32/64/128/256 va unlabeled
   proxy objective.
4. Bao cao server FLOPs, latency, memory va communication; so sanh o matched
   wall-clock budget.
5. Tang len it nhat 10 seeds hoac bootstrap theo run truoc final thesis claim.
6. Chuyen empirical gate thanh one-sided confidence or anytime-valid gate neu
   muon claim safety co statistical guarantee.

## Literature anchors

- AdaLoRA: https://arxiv.org/abs/2303.10512
- FedEx-LoRA: https://arxiv.org/abs/2410.09432
- FedRot-LoRA: https://arxiv.org/abs/2602.23638
- FLoRG: https://arxiv.org/abs/2602.17095
- GLoRA: https://arxiv.org/abs/2605.06733
- AlignFed: https://arxiv.org/abs/2606.08197
- FSLoRA: https://arxiv.org/abs/2501.19389
- SDFLoRA: https://arxiv.org/abs/2601.11219
