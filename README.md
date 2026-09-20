# Exp2: 공동 학습 기반 masked expression + 공간 메시지

receiver의 남은 발현으로 예측할 수 있는 범위를 넘어, 실제 이웃의 발현이 masked gene 예측에 기여하는지 측정한다. `internal + external`을 처음부터 하나의 Huber reconstruction objective로 공동 학습한다. 별도 intrinsic 사전학습, freeze, detach, branch 단독 reconstruction loss는 없다.

이 구현의 성공 범위는 **추가적인 공간 정보의 예측 기여**다. 두 출력은 모델상 기여이며, 생물학적 인과 효과나 cell-autonomous 발현량의 식별을 보장하지 않는다. 일반화 평가는 이번 범위에 포함하지 않는다.

## Ready2Start — 새 서버에서 복사해서 실행

Ubuntu/Debian, Python 3.10 이상 기준이다. 아래 블록 하나로 **코드와 포함된 FISH 데이터 clone → 환경 설치 → 데이터 검증 → 테스트 → 실제 데이터 smoke 학습/평가 → 본 학습/평가**까지 실행한다. 기존 `experiment1` 폴더나 수동 데이터 복사는 필요 없다. Python 패키지 설치는 인터넷 연결이 필요하며 NVIDIA driver는 GPU 서버에 이미 설치되어 있어야 한다.

저장소 주소는 `https://github.com/kevinkai1220/experiment2.git`이다. private 저장소라면 clone 전에 해당 저장소에 접근할 수 있는 GitHub 인증을 설정한다.

```bash
# 아래 전체 블록을 새 서버의 bash 터미널에 붙여넣기
bash <<'EXP2_SETUP'
set -euo pipefail

# 1. 시스템 도구 설치 (Ubuntu/Debian, sudo 권한 필요)
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip ca-certificates

# 2. 코드와 포함된 FISH 데이터 받기
git clone https://github.com/kevinkai1220/experiment2.git
cd experiment2

# 3. 독립 Python 환경 및 라이브러리 설치
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# 4. clone에 포함된 FISH 데이터 + 좌표의 SHA-256/형식/metadata 검증
# 파일: data/fish_cancer_raw.h5ad, data/fish_coordinates.csv
python scripts/prepare_fish.py

# 5. 공동 학습/누수 방지/체크포인트 자동 테스트
python -m pytest -q

# 6. 실제 FISH 데이터로 빠른 CPU 실행 확인 (1 epoch, 작은 kNN 모델)
# all/learned 모드는 위 pytest에서도 별도로 학습 및 checkpoint 검증
python main.py train --config configs/fish.json \
  --epochs 1 --hidden 16 --graph-mode knn --neighbors 4 --shuffle-repeats 1 \
  --output results/fish_smoke --device cpu
python main.py evaluate --checkpoint results/fish_smoke/checkpoint.pt \
  --output results/fish_smoke/evaluation --device cpu

# 7. 본 학습: 모든 같은-tissue cell 사용, 고정 p=1, 기본 50 epochs
# CUDA 사용 가능하면 GPU, 없으면 CPU (all 모드 CPU 본 학습은 오래 걸림)
python main.py train --config configs/fish.json \
  --output results/fish_joint --device auto

# 8. best checkpoint 평가 + masked-expression 예측 저장
python main.py evaluate --checkpoint results/fish_joint/checkpoint.pt \
  --output results/fish_joint/evaluation --device auto
EXP2_SETUP
```

GPU 사용을 필수로 하려면 7·8번의 `--device auto`를 `--device cuda`로 바꾼다. CUDA 사용이 불가능하면 즉시 오류가 난다. GPU 본 학습 대신 실행 확인까지만 하려면 7·8번을 제외한다. NVIDIA GPU가 있는데도 `CUDA available: False`이면 서버 driver와 PyTorch 빌드의 호환성을 먼저 확인한다. 위 명령은 NVIDIA driver 자체를 설치하지 않는다.

**이미 clone한 경우** 해당 `experiment2` 폴더에서 아래를 실행하면 된다. 시스템의 Python/venv가 준비되어 있다고 가정한다.

```bash
bash <<'EXP2_RUN'
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/prepare_fish.py
python -m pytest -q
python main.py train --config configs/fish.json --output results/fish_joint --device auto
python main.py evaluate --checkpoint results/fish_joint/checkpoint.pt \
  --output results/fish_joint/evaluation --device auto
EXP2_RUN
```

저장소에 포함한 데이터는 전체 Perturb-FISH 원본 이미지가 아니라 **이 실험에서 검증한 전처리된 cancer cohort**(7,496 cells × 500 raw-count genes, 약 1.8 MB의 h5ad/좌표 2개 파일)다. cell type·perturbation metadata는 h5ad에 포함되어 있다. 별도 전처리나 Exp1 split 생성 없이 이 전체 cohort를 입력으로 사용한다. expression은 raw counts이고 좌표는 같은 cell 순서다. `prepare_fish.py`는 포함된 파일의 SHA-256과 형식을 확인하며, 파일을 덮어쓰거나 외부 URL에서 데이터를 받지 않는다. 출처·버전은 [data/README.md](data/README.md)에 기록했다. 검증에는 표본의 물리적 단일 조직 여부를 입증하는 과정이 포함되지 않으므로 아래 데이터 계약의 단일 조직 가정도 확인한다.

**GitHub에 게시할 때 `data/fish_cancer_raw.h5ad`와 `data/fish_coordinates.csv`도 반드시 함께 commit한다.** 두 파일은 `.gitignore` 대상이 아니며 Git LFS 없이 저장할 수 있다. 코드만 게시하면 새 서버에서 데이터 검증 단계가 실패한다.

CUDA 학습과 생물학적 성공은 로컬 CPU smoke test로 검증할 수 없다. 초기 작은 실행에서 진단 실패가 나더라도 이를 숨기거나 성공으로 바꾸지 않는다. 학습/평가 결과가 있는 디렉터리는 덮어쓰지 않으므로 새 `--output`을 사용한다. 현재 CLI는 새 학습과 checkpoint 평가를 지원하며 optimizer resume은 지원하지 않는다.

## 데이터 계약과 eligible_receiver_mask

입력은 하나의 AnnData `.h5ad`다.

- `X`: 유한한 비음수 continuous expression/count, `[cells, genes]`. 기본은 raw counts에 `log1p`; `--transform raw`는 변환하지 않는다. 이미 log 변환했다면 재변환을 피하도록 `raw`를 명시한다.
- `obs['celltype2']`: receiver/sender cell type. 결측은 오류. `--type-key`로 변경한다.
- `obs['perturbation']`: `Control`만 supervised receiver로 선정. `--perturbation-key`, `--control-label`로 변경한다. 결측은 `__unknown__` sender로 남고 receiver 감독에서 제외된다. `n_perturb` 숫자로 판정하지 않는다.
- `obsm['spatial']`: row-aligned 좌표. 또는 `--coordinates`로 헤더 없는 숫자 CSV를 지정한다. CSV 정렬은 h5ad cell 순서와 정확히 같아야 한다.
- 물리적 조직/절편 ID: `--tissue-key sample_id`를 권장한다. 모든 graph edge는 같은 tissue 안에서만 생성된다. 해당 정보가 없으면 명시적 `--assume-single-tissue`가 필요하다.
- cell ID와 gene 이름은 중복을 허용하지 않는다. checkpoint 평가는 원래의 cell/gene 순서와 type vocabulary를 요구한다.

```python
eligible_receiver_mask = (obs[perturbation_key] == control_label)
```

`Control`이 해당 데이터에서 실제로 비표적/비교란 cell을 뜻하는지 확인해야 한다. 이웃에 perturbed cell이 있더라도 Control receiver는 제외하지 않는다. perturbed 및 unknown cell은 sender로 쓸 수 있다. 별도 control 조직도 같은 파일에 넣되 tissue ID를 구분하고 동일 모델·동일 objective를 사용한다.

현재 예제 원본은 7,496 cells × 500 genes, `cancer` 한 cell type, Control 3,452개 및 알려진 perturbation 4,044개다. tissue ID가 없으므로 `configs/fish.json`은 단일 조직을 **가정**한다. 이는 검증된 조직 식별이 아니다. 서로 다른 표본의 좌표를 합친 데이터라면 해당 가정으로 본 실험을 진행하지 말고 tissue ID를 제공해야 한다.

기존 Exp1의 train/test 파일 대신 전체 원본을 사용한다. Exp2 안에서 eligible receiver를 type/tissue별 80/20으로 나누며 모든 cell의 공간 context를 보존한다. 검증 receiver의 모든 gene은 **학습 forward에서 sender로 사용될 때도 mask**한다. 검증에서는 그 receiver의 관측 가능한 나머지 gene을 사용한다. 이는 같은 조직 내 receiver holdout이고, sender pool을 공유하는 transductive 평가다. 새로운 조직/perturbation으로의 일반화 주장을 하지 않는다.

## 모델 및 학습

```text
corrupted receiver ── InternalEncoder ── h ── InternalDecoder_type ── internal
                                        │
corrupted senders ── MessageEncoder_type ─┼─ weighted one-hop message M
                                        │
                         F_type(h,M,g) - F_type(h,0,g) ────────────── delta

                         prediction = internal + delta
```

각 gene token은 `gene_embedding + ValueMLP(value)` 또는 `gene_embedding + MASK_embedding`이다. encoder는 nonlinear token MLP → mean pooling → LayerNorm/MLP로 구성한다. encoder 입력에는 cell ID, 좌표, 이웃 metadata를 넣지 않는다. sender encoder는 cell type별 독립 모듈이며 이웃 메시지를 다시 받지 않는다. internal decoder와 external decoder도 receiver type별이다. 최종 delta와 prediction에는 ReLU를 적용하지 않는다.

한 forward에서 receiver와 모든 real/null sender의 cell ID를 합쳐 중복 없는 node 목록을 만들고, cell마다 하나의 corrupted vector와 boolean mask만 만든다. 모든 encoder는 이 tensor를 공유한다. masked 값은 ValueMLP 전에도 0으로 지워 방어한다. 원래 expression은 corruption 생성 경계, training-only module 계산, 정답 loss/evaluation에서만 읽는다. 모델 forward에는 original tensor를 전달하지 않는다.

graph는 아래 두 mode를 지원한다. **FISH 예제 설정의 기본은 `all`**이며, 별도 config가 없는 CLI의 기본은 기존 `knn`을 유지한다. 모든 모드에서 다른 tissue 연결과 self-edge는 없다. 좌표 중복도 처리하며, 고립 cell은 `M=0`, `delta=0`이 된다.

## 모든 cell / kNN 및 고정 / 학습 가능한 거리 감쇠

| 옵션 | 동작 |
|---|---|
| `--graph-mode all` | 같은 물리적 tissue의 모든 다른 cell 사용. k 제한 없음. `--neighbors`는 무시하며 radius cutoff는 허용하지 않음 |
| `--graph-mode knn --neighbors 16` | 기존 tissue별 kNN. 선택적으로 `--radius` 지정 가능 |
| `--distance-mode fixed --distance-power 1` | 고정된 역거리 감쇠 지수 p |
| `--distance-mode learned --distance-power 1` | p=1에서 시작해 공동 reconstruction/auxiliary gradient로 p를 학습 |

두 축은 독립적이어서 네 조합 모두 가능하다. 모든 cell 사용은 다른 조직의 cell까지 섞는다는 의미가 아니다.

```text
raw_weight_ij = (distance_ij + epsilon)^(-p)
alpha_ij = raw_weight_ij / sum_j(raw_weight_ij)
M_i = sum_j(alpha_ij * message_j)

learned mode: p = softplus(raw_p) + 0.0001
```

기본 epsilon은 0.001이며 좌표 단위에 맞춰 조정할 수 있다. learned p는 전체 모델에 하나인 양의 scalar다. p가 커지면 가까운 cell에 더 집중하고, 0에 가까워지면 전체 cell이 거의 같은 가중치를 갖는다. 거리와 무관한 자유로운 edge별 attention을 학습하는 방식은 아니다. p와 그 gradient norm을 매 epoch 기록하고 checkpoint에도 저장한다.

가중치는 **receiver별 합이 1**이므로 먼 cell의 상대적 기여는 작아지지만, 모든 cell이 멀어진다는 이유만으로 전체 message 크기가 반드시 작아지는 것은 아니다. 이는 원안의 normalized aggregation을 유지한 선택이다. 절대 거리 증가에 따라 message 전체 크기도 줄이는 unnormalized mode는 이번 구현에 포함하지 않았다.

```bash
# 모든 cell + 고정 inverse-distance (FISH 기본 설정)
python main.py train --config configs/fish.json --graph-mode all \
  --distance-mode fixed --distance-power 1 --output results/all_fixed --device cuda

# 모든 cell + 학습 가능한 거리 감쇠
python main.py train --config configs/fish.json --graph-mode all \
  --distance-mode learned --distance-power 1 --output results/all_learned --device cuda

# 기존 kNN + 학습 가능한 거리 감쇠
python main.py train --config configs/fish.json --graph-mode knn --neighbors 16 \
  --distance-mode learned --output results/knn_learned --device cuda

# 데이터 없이 all + learned 모드를 CPU에서 확인
python main.py train --synthetic --epochs 1 --hidden 16 --graph-mode all \
  --distance-mode learned --checkpoint-senders --sender-chunk-size 16 \
  --shuffle-repeats 1 --output results/all_learned_smoke --device cpu
python main.py evaluate --checkpoint results/all_learned_smoke/checkpoint.pt \
  --output results/all_learned_smoke/evaluation --device cpu
```

all mode는 graph/null graph를 row별로 지연 생성해 전체 N×N distance/weight 행렬을 메모리에 보관하지 않는다. sender encoder는 `--sender-chunk-size` 단위로 실행한다. FISH config는 32-cell chunk와 `--checkpoint-senders`를 사용해 backward에서 activation을 재계산하며 모든 sender의 gradient를 유지한다. 메모리는 줄지만 계산 시간이 증가한다. 학습 중 sender를 freeze/detach/cache하지 않는다. all mode는 모든 sender를 매 training batch에서 다시 encode하므로 kNN보다 상당히 느릴 수 있다.

validation에서는 sender mask가 고정되고 weight 업데이트가 없으므로 sender encoding을 한 번 계산해 재사용한다. receiver의 mask가 다른 cached row는 실제 corrupted receiver 입력으로 재계산하며, 어떤 real/null edge도 동시에 평가하는 receiver의 sender row를 사용하지 않는다. all mode에서는 같은 tissue의 평가 receiver batch가 사실상 1개씩 처리된다. 이 조치는 원래 masked target이 cache를 통해 새는 것을 방지하면서 sender 입력을 고정한다.

학습 가능한 p를 쓰더라도 auxiliary q의 거리 가중치는 **초기 p로 고정**한다. 학습되는 parameter에 따라 감독 target까지 움직이는 문제를 피하려는 결정이며, 실제 reconstruction message에는 매번 현재 학습된 p를 적용한다.

```text
L = Huber(prediction[masked eligible positions], target)
    + lambda_msg * MSE(EnvironmentDecoder(M), standardized_q)
    + lambda_adv * MSE(EnvironmentPredictor(GRL(h), receiver_type), standardized_q)
```

환경 predictor에 receiver type one-hot을 직접 주고, `h`에만 GRL을 적용한다. predictor는 정상 방향으로 MSE를 줄이고 intrinsic encoder는 반대 gradient를 받는다. GRL 계수는 1, `lambda_adv=0.01`을 loss에 한 번만 곱한다. `lambda_msg=0.05`. 환경 구성비/거리 feature는 train receiver 기준으로 표준화하고 MSE로 회귀한다. 합이 1인 조성 출력이나 확률로 강제하지 않는 auxiliary representation objective다.

`q`는 거리 가중 이웃 cell-type 구성, perturbation-label 구성(unknown 포함), 같은 tissue 내 가장 가까운 알려진 perturbed cell까지의 `log1p` 거리, tissue 내 perturbed cell 유무, 이웃 유무로 구성한다. perturbed cell이 없으면 거리 0과 별도 유무 flag로 표현한다. label 구성에서 perturbed fraction도 도출할 수 있다. q는 auxiliary target만 제공하며 reconstruction decoder의 직접 입력이 아니다. type conditioning은 유용한 receiver state가 adversary에 의해 손상될 위험을 완전히 없애지는 않는다.

## 마스킹 및 비교 실험

기본 train batch mixture는 random 20% 50회 비중, high 40–50% 25회 비중, module 25회 비중이다. 이는 배치별 확률 `(0.5, 0.25, 0.25)`이며 정확한 epoch별 개수는 달라진다.

module은 **training Control receiver만** 사용해 type별 평균을 제거하고 gene correlation을 구한다. 거리 `1-correlation`, average-linkage hierarchical clustering으로 최대 20개 module을 만든다. mask는 module 전체를 선택해 목표 최소 40% gene에 도달할 때까지 추가한다. 마지막 module 때문에 실제 비율이 더 높을 수 있어 실제 비율도 보고한다. singleton·상수 gene도 처리한다. signed correlation을 사용하므로 반대 방향으로 반응하는 gene들이 반드시 같은 module이 되는 것은 아니다. pathway module은 아직 구현하지 않았다.

sender 기본 mask 비율은 receiver 비율과 독립적인 20%다. train에서는 같은 batch의 receiver가 다른 receiver의 sender이면 그 cell의 receiver mask가 우선하며, 복수 corrupted 표현을 만들지 않는다. validation에서는 real/null graph 모두에서 **동일 배치 receiver끼리 서로 sender가 되지 않도록 배치를 구성**한다. sender mask는 epoch·receiver mask 조건 간 완전히 고정한다. 따라서 20/40/50/60% 및 module 평가에서 receiver 정보만 줄이는 비교가 가능하다. 배치 크기는 이 제약에 따라 줄어들 수 있다.

```bash
# mixture, random20/40/50/60, module 각각 aux on/off × seed 42/43/44 = 36 runs
# GPU 서버의 경로를 configs/fish.json 또는 별도 JSON에 반영한다.
python sweep.py --config configs/fish.json --output results/ablations --device cuda --dry-run
python sweep.py --config configs/fish.json --output results/ablations --device cuda

# sender corruption 자체의 효과는 별도 실험 축
python main.py train --config configs/fish.json --sender-ratio 0.5 \
  --output results/sender50 --device cuda

# 두 auxiliary를 제거해도 같은 공동 reconstruction 학습을 유지
python main.py train --config configs/fish.json --lambda-msg 0 --lambda-adv 0 \
  --output results/no_aux --device cuda
```

## 매 validation epoch의 진단과 결과 해석

동일 mask 위치에서 Huber/MSE/MAE를 모두 보고한다.

| 평가 | 실제 조작 |
|---|---|
| `full_model`, `real_messages` | 같은 실제 full prediction의 두 이름 |
| `internal_only` | 이미 공동 학습된 internal 출력만 읽는 진단. 별도 학습하지 않음 |
| `shuffled_messages` | tissue 내 sender ID permutation으로 message를 edge slot에 재할당, 가중치 보존 |
| `conditional_shuffled_messages` | tissue·sender type·perturbation label 내에서만 동일 조작 |
| `shuffled_neighborhoods` | receiver별 같은 tissue의 무작위 sender 집합으로 rewiring; edge 수·가중치 보존 |

shuffle은 self-edge가 생기면 그 slot의 원래 sender로 되돌린다. 이 때문에 엄밀한 순열이 아니며 sender 중복이 생길 수 있다. 검증 receiver에서 변경된 edge 비율을 보고한다. rewiring은 중복 없이 샘플링하지만 원래 거리 분포를 다시 계산하지 않는다. 원래 weight slot을 고정한 공간 대응 파괴 대조군이며 물리적 공간 생성 모델은 아니다. **all mode에서는 이미 모든 cell이 연결되어 있으므로 neighborhood의 집합 자체는 바뀌지 않고 sender와 거리 weight의 대응만 바뀐다.** 따라서 이를 별개의 이웃 집합 효과로 해석하지 않는다. default 3회 shuffle의 **loss를 평균**하고, 예측을 평균해 ensemble하지 않는다. type/perturbation strata가 작으면 conditional null의 검정력이 부족할 수 있다.

`internal_only`는 최적화된 receiver-only baseline이 아니다. 이 비교만으로 모든 가능한 receiver-only predictor보다 우수하다고 주장할 수 없다. 실제/조건부 shuffle 비교를 함께 사용하고, 별도의 receiver-only reconstruction 모델은 학습하지 않는다.

masked target 위치에서 delta의 평균 절대값·표준편차, internal/delta 분산, delta RMS / target RMS도 출력한다. training total loss의 encoder/message/external gradient norm 평균을 매 validation 보고서에 기록한다. auxiliary만 gradient를 만드는 착시를 확인할 수 있도록 epoch의 첫 training batch에서 **reconstruction-only gradient norm**도 별도 기록한다. validation data로 optimizer를 업데이트하지 않는다.

기본 실패 기준은 다음과 같고 config로 변경 가능하다.

- full의 Huber 개선이 internal-only 대비 `max(0.0001, 1% × 비교 loss)` 이하.
- real message의 개선이 global shuffle 대비 같은 기준 이하.
- real message의 개선이 conditional shuffle 대비 같은 기준 이하: sender state의 추가 기여 실패로 별도 표시.
- delta RMS / target RMS가 `0.001` 미만.

어느 하나라도 해당하면 해당 mask 조건은 `FAIL_DIAGNOSTIC`이다. 모두 통과해도 `PASS_DIAGNOSTICS_NOT_CAUSAL_PROOF`일 뿐이다. 이 기준은 초기 실용적 기준이며 임계값의 과학적 타당성을 보장하지 않는다. 실험 결과를 보고 유리하게 바꾸지 않는다.

같은 tissue를 독립 표본으로 취급한 paired shuffle gain을 보고한다. tissue가 둘 이상이면 tissue bootstrap 95% 구간을 제공하며, 하나이면 `null`이다. 같은 조직의 cell을 독립 생물학적 반복으로 간주한 신뢰구간은 제공하지 않는다. 여러 seed의 반복은 최적화/분할 민감도이지 독립 조직 반복이 아니다. 검증 데이터로 checkpoint를 선택하므로 최종 독립 test 성능으로 부르지 않는다.

## 결과 파일

- `config.json`: 전체 실행 설정.
- `data_report.json`: eligible 정의/개수, type/tissue별 개수, sender 구성, 실제 device와 평가 범위.
- `split.npz`: train/validation indices, `eligible_receiver_mask`, cell IDs.
- `modules.json`: training-only module별 gene 이름.
- `history.jsonl`, `validation_latest.json`: 매 epoch loss, gradient norm, 모든 mask 조건의 collapse 진단.
- `checkpoint.pt`: validation mask 조건별 full Huber의 산술평균으로 선택한 best 모델, modules, q 정규화 통계, split, gene/type schema.
- `evaluation/evaluation.json`, `predictions.npz`: checkpoint 재평가 및 각 mask 조건의 internal/delta/prediction/mask. 배열은 `[validation receivers, genes]`; 평가는 mask가 true인 위치만 유효하며 gene/cell ID가 함께 저장된다.
- sweep의 `summary.json`: 완료한 모든 조건/seed의 best checkpoint 결과.

checkpoint에 원본 expression은 넣지 않는다. 재평가에는 동일 데이터와 metadata가 필요하다. 예측값은 기본 `log1p` 단위이며 `expm1(internal) + expm1(delta)`는 raw-scale prediction이 아니다. raw 단위 평가가 필요하면 `--transform raw` 실험을 별도로 수행한다.

## 원안에서 수정·변경·확장한 사항

이 섹션은 사용자 A1–A6 결정과 이후 구현 판단을 구분해 기록한다.

1. **주장 범위 — A1:** 생물학적 인과 분해 대신 receiver 정보에 추가되는 공간 정보의 예측 기여를 주 질문으로 확정했다. additive 출력은 여전히 임의의 receiver-dependent 함수가 두 branch 사이에서 이동할 수 있으므로 해석에 주의가 필요하다.
2. **메시지 의존 구조 — A2:** 원래 `D_external(h,M,g)`를 `F(h,M,g)-F(h,0,g)`로 변경했다. delta가 receiver 입력만으로 메시지 없이 나오는 경로를 제거한다. 기준점은 zero latent message이며 평균 control 환경이나 실제 반사실 상태를 뜻하지 않는다. 두 항 모두 공동 학습하고, branch dropout reconstruction loss는 추가하지 않는다. 모델이 메시지를 무시하거나 상수처럼 이용하는 경우까지 방지하지는 못하므로 shuffle 진단을 유지한다.
3. **평가 범위 — A3:** 새로운 조직·perturbation 일반화는 보류했다. 같은 tissue receiver holdout을 구현하고, 학습 때 validation receiver의 발현 전체를 sender 입력에서도 가렸다. 이는 sender pool을 공유하므로 엄격한 inductive split은 아니다.
4. **대조군 — A4:** global shuffle과 type/perturbation 조건부 shuffle을 모두 추가했다. 전자는 이웃 구성 포함 전체 정보, 후자는 해당 구성을 넘어선 sender state 정보를 점검한다. 별도 neighborhood rewiring도 유지한다.
5. **adversary — A5:** receiver type을 조건으로 predictor에 제공하고 `h`의 추가 환경 정보를 억제하도록 변경했다. 환경 loss는 training receiver 통계로 표준화한 feature별 MSE를 사용한다. aux on/off를 sweep에 포함했다.
6. **mask 축 분리 — A6:** sender ratio를 별도 config로 분리했다. validation sender mask를 고정하고 receiver 간 메시지 edge가 없는 평가 배치를 구성해 receiver corruption만 변화시킨다. training의 receiver/sender 중첩에서는 동일 corrupted tensor 규칙을 우선한다.
7. **발현 scale — 구현 결정:** 기본을 gene별 `log1p`로 정했다. masked gene이 cell별 library-size 정규화를 통해 다른 gene 값에 유출되는 경로를 피하고 large count의 영향을 완화한다. 따라서 additive 의미도 기본적으로 log scale이다. raw regression 옵션을 제공한다.
8. **모델 용량 — 구현 결정:** gene token nonlinear pooling encoder를 채택했다. Transformer encoder는 필수 조건이 아니며 CPU 검증과 작은 공간 패널의 출발점에 맞춘 선택이다. 복잡한 gene interaction 표현력은 별도 모델 용량 실험이 필요하다.
9. **이웃 정의 — 사용자 추가 요청 반영:** 초기 구현의 kNN + 선택적 radius에 더해 같은 tissue의 모든 다른 cell을 사용하는 `all` mode를 추가했고 FISH config의 기본으로 설정했다. 원래 distance weighting, one-hop, no-self 조건은 유지한다. 추가로 양의 전역 감쇠 지수 p를 공동 학습하는 `learned` mode를 제공한다. normalized aggregation이므로 절대 signal 감쇠와는 구분한다.
10. **module 구성 — 구현 결정:** training Control cell에서 type별 평균을 제거한 signed correlation을 사용한다. type 혼합에 따른 module 형성을 줄이려는 선택이며 원안의 train-only 원칙을 유지한다. module 전체 mask로 목표 비율을 초과할 수 있다.
11. **판정 및 기록 — 구현 결정:** configurable 수치 실패 기준, conditional null 실패 flag, 반복 shuffle, 실제 변경 edge 비율, reconstruction-only gradient 점검을 추가했다. seed sweep 및 tissue 단위 불확실성도 구현했다. 이 추가 항목은 reconstruction 학습 loss를 바꾸지 않는다.
12. **실데이터 가정 — 구현 결정:** 기존 데이터에 tissue ID가 없어 예제 config에서 단일 조직 가정을 명시한다. 데이터가 실제 한 조직인지 새로 입증하지 않았으며, 한 가지 cell type뿐인 현재 데이터로 여러 type 간 반응 차이를 검증했다고 주장하지 않는다.
13. **전체 cell 실행 효율 — 구현 결정:** lazy graph, sender chunking 및 gradient checkpointing을 추가했다. validation에 한해 고정 corruption의 sender representation을 재사용하며 학습에는 적용하지 않는다. q target은 초기 distance power 기준으로 고정한다. all mode에서 shuffled neighborhood는 연결 집합이 아닌 distance-weight 대응의 교란임을 보고한다.

## 로컬 검증

실행 기록은 `VALIDATION.md`에 정리한다. 테스트는 공동 reconstruction gradient, masked-value 누수 차단, zero-message delta, GRL 방향, no-self/tissue graph, training-only modules, eligible split, 고립 cell, mask 강도별 sender 입력 고정, collapse 판정, checkpoint round trip을 검사한다.
