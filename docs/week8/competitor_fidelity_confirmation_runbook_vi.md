# Competitor fidelity va confirmation-only runbook

Cap nhat 2026-09-11: da them Spectral post-hoc va AlignFed reference runner.
Xem [audit implementation moi](paper_baseline_implementation_audit_vi.md).
Bang duoi day mo ta confirmation v3 cu; code moi khong nang fidelity cua ket qua cu.

Ngay: 2026-09-09. Matrix: `configs/rift_core_heldout_confirmation_matrix.json`
(v3). Khong dung cac seed 6101-6106 de tune tham so sau khi da xem ket qua.

## Ket luan audit implementation

| Ten trong confirmation | Muc fidelity | Co duoc goi full paper? |
|---|---|---|
| Raw async | Exact matched control cua simulator | Khong phai paper method |
| Freshness | Exponential staleness weighting | Control, khong phai mot reproduction cu the |
| FedRot | Procrustes, `det(R)>0`, hard rotation, luan phien A/B | Khong; day la paper-faithful operator port vao immediate async |
| Spectral Surgery | Mean per-example `abs(u^T G v)`, smooth reweight, L1 mass preservation, fixed paper factors | Khong; day la paper-faithful operator port vao moi async return |
| AlignFed calibration | Whole-update calibration gate | Khong phai AlignFed day du |
| RIFT | Positive component filter + gate | Internal ablation |
| RIFT-Diag | Equal-budget diagonal repair + gate | Internal ablation |
| RIFT-Core | Full-core repair + gate | Proposed method |

FedRot port da doi chieu official code commit
`36a67ef205818c292cf9ff361a80f969de16cedd`. Khac paper protocol o cho
official FedRot dong bo nhieu client trong mot round; confirmation nhan tung
return async va noi suy voi server state hien tai.

Spectral Surgery da thay `spectral_filter` trong confirmation. Port moi giu
U/V, tinh magnitude sensitivity truoc khi average, chuan hoa trong moi module,
dung `smooth_abs` voi core/noise 0.2, amp 1.25, suppress 0.8, temperature
0.35 va bao toan tong singular value L1. De fair voi RIFT, no dung cung 24
calibration examples, cung `q_proj/v_proj`, cung class-NLL va cung incoming
innovation. Paper goc mac dinh 128 examples, edit `o_proj/down_proj` va edit
mot adapter da train xong. Vi vay ten dung khi bao cao la
"Spectral Surgery async operator port", khong phai full reproduction.

`alignfed_calibration` khong duoc doi ten thanh AlignFed. Full AlignFed can
buffer trigger, group theo version, intra-group centering, representation
regularization khi local train, hoc linear transform tren penultimate features,
freshness, norm va participation weighting. Voi `buffer_size=1`, group thuong
chi co mot update va centering se dua update ve zero. Paper cung chua cung cap
official code/chi tiet du de port transformation mot cach kiem chung.

FedEx khong nam trong confirmation v3. Alias `fedex` cu chi cong innovation
trong product space roi recompress rank; no khong phai FedEx-LoRA. Full
FedEx-LoRA dong bo mot cohort, average A/B, tinh residual giua mean product va
product cua means, sau do cong residual vao frozen `W0` cho round ke tiep.
Dieu nay khong tuong duong immediate async va khong phu hop backbone 4-bit bat
bien hien tai. Official source da doi chieu tai commit
`2fa2e4a243f93c829e2b601e459e45aa748e4149`.

## Acc cua doi thu co the cao hon khong?

Co. Baseline cu co the danh gia thap FedRot va Spectral Surgery vi thieu cac
thanh phan tren. Port v3 co the tang hoac giam accuracy tuy task; Spectral
Surgery paper cung cho thay policy gradient co tinh task-dependent. Vi chua
chay seed confirmation, khong duoc khang dinh truoc rang RIFT se thang.

Full AlignFed/FedEx cung co the cao hon proxy, nhung chay chung ten trong
immediate-async matrix se khong phai so sanh cung protocol. Neu thesis claim
thang implementation official end-to-end, can mot benchmark cohort/buffer
rieng; ket qua do khong duoc tron vao verdict v3 nay.

## Preflight confirmation

Phai commit code truoc. Runner chi resume result dung exact commit, schema,
matrix hash va config hash; worktree dirty se khong duoc reuse.

```powershell
python -m pytest -q
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --output-root outputs/rift_core_confirmation_v3_1_5b --dry-run
```

Expected: 4 tasks x 1 regime x 8 methods x 6 seeds = 192 jobs. Khong chay
development matrix va khong dung `--force` khi resume.

## Chay confirmation

Chay tung task de de resume va theo doi:

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task sst2 --output-root outputs/rift_core_confirmation_v3_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task qnli --output-root outputs/rift_core_confirmation_v3_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task mnli_m --output-root outputs/rift_core_confirmation_v3_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task mnli_mm --output-root outputs/rift_core_confirmation_v3_1_5b
```

Tren model 3B, dung output root rieng. Khong truyen revision 1.5B cho model
3B; neu co exact Qwen 3B commit thi pin bang `--model-revision`.

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --model-name Qwen/Qwen2.5-3B-Instruct --output-root outputs/rift_core_confirmation_v3_3b --dry-run
```

Bo `--dry-run` sau khi kiem tra GPU/duong dan. Lap lai cung command se resume
cac job hop le da xong.

## Tong hop verdict

```powershell
python scripts/analyze_kaggle_3b_rift_competitors.py --input-dir outputs/rift_core_confirmation_v3_1_5b --output-dir outputs/rift_core_confirmation_v3_1_5b_analysis --matrix configs/rift_core_heldout_confirmation_matrix.json --target rift_core
```

Chi doc verdict khi completeness/provenance deu pass. Primary quality la
paired final Accuracy va class NLL tren ca 6 seed. Safety la harmful rate,
late harmful rate, cumulative/worst late harm; kem acceptance, client coverage,
runtime va peak memory. Bao cao moi seed; khong bo seed sau khi xem ket qua.
