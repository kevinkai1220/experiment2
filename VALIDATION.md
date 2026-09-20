# CPU 실행 검증 기록

검증 날짜: 2026-09-20. 사용 환경: Python 3.13, PyTorch 2.14.0+cpu, 기존 `../experiment1/.venv/bin/python`. 이 환경을 재사용했으며 Exp1 코드/데이터를 수정하지 않았다.

## 자동 테스트

```bash
../experiment1/.venv/bin/python -m pytest -q
```

결과: **11 passed**. 공동 reconstruction만으로 양 branch에 gradient가 전달되는지, masked 값 차단, 메시지 0일 때 delta 0, signed delta, GRL gradient 방향, 공간 graph와 null의 tissue/no-self 조건, training-only module, receiver eligibility, 고립 cell, validation sender corruption 고정, collapse 판정, checkpoint 저장/재로딩을 검사했다.

`python -m compileall -q exp2 main.py sweep.py tests`도 통과했다. `sweep.py --config configs/fish.json --dry-run --device cpu`는 36개의 실행 설정을 생성했다. 36개 전체 학습은 수행하지 않았다.

## 실제 FISH 데이터 smoke run

```bash
../experiment1/.venv/bin/python main.py train --config configs/fish.json \
  --epochs 1 --hidden 16 --batch-size 16 --neighbors 4 --shuffle-repeats 1 \
  --output results/fish_cpu_verified --device cpu
```

- 전체: 7,496 cells × 500 genes.
- eligible receiver: `perturbation == 'Control'`, 3,452개.
- train receiver: 2,762개; validation receiver: 690개.
- known perturbed sender: 4,044개; unknown label sender: 0개.
- cell type: `cancer` 1종; tissue ID 부재로 단일 조직 가정.
- 1 epoch의 train masked Huber: 0.094116.
- validation 5개 mask 조건 full Huber 산술평균: 0.062993.

| Validation mask | Internal only | Real messages | Shuffled messages | Conditional shuffle |
|---|---:|---:|---:|---:|
| Random 20% | 0.080590 | 0.061001 | 0.061008 | 0.061008 |
| Random 40% | 0.079004 | 0.060802 | 0.060810 | 0.060811 |
| Random 50% | 0.078764 | 0.061335 | 0.061343 | 0.061345 |
| Random 60% | 0.078561 | 0.061662 | 0.061671 | 0.061672 |
| Module | 0.088721 | 0.070167 | 0.070179 | 0.070181 |

숫자는 기본 log1p expression 단위의 Huber다. 실제 메시지는 internal-only보다 개선됐지만, shuffle과의 차이가 사전 기준에 못 미쳤다. **모든 조건이 `FAIL_DIAGNOSTIC`이며 공간 기여 실험 성공으로 인정하지 않는다.** delta가 단순히 0인 경우는 아니었다. 짧은 작은 모델 실행이므로 이것을 최종 모델의 성공/실패 결론으로 일반화하지 않는다.

첫 training batch에서 auxiliary 없이 reconstruction 자체가 주는 gradient norm:

| Component | Norm |
|---|---:|
| Internal encoder | 0.078770 |
| Message encoder | 0.075826 |
| External decoder | 0.068842 |

이는 두 branch가 공동 reconstruction에 연결되어 있음을 보여주는 실행 진단이며, 유용한 생물학적 분해의 증거가 아니다.

## 재평가 명령

```bash
../experiment1/.venv/bin/python main.py evaluate \
  --checkpoint results/fish_cpu_verified/checkpoint.pt \
  --output results/fish_cpu_verified/evaluation --device cpu
```

재평가는 저장된 module, q 통계, split, model weight와 고정된 validation masks/nulls를 사용한다. 결과는 `results/fish_cpu_verified/evaluation/evaluation.json` 및 `predictions.npz`에 저장된다.

실행 완료 후 모든 mask/대조군의 Huber·MSE·MAE를 저장된 training validation 결과와 비교했다. 최대 차이는 **0.0**이었다. 예측 배열은 690 receiver × 500 genes이며, 각 조건의 `prediction == internal + delta`도 확인했다.

## 검증하지 않은 범위

GPU/CUDA 실행, 50 epoch 본 학습, 전체 ablation sweep, 독립 조직 일반화, 생물학적 인과 분해는 수행하지 않았다. 현재 단일 조직 가정의 물리적 정확성도 입증하지 않았다. GPU 서버에서 원래 설정의 충분한 학습과 데이터 메타데이터 확인이 필요하다.

## 추가 검증: 저장소 포함 데이터, all-cell 및 learned-distance 모드

위 11개 테스트 및 1 epoch 표는 초기 kNN 구현의 기록이다. 이후 `configs/fish.json`은 저장소 내부 `data/`를 사용하고 기본 graph mode가 `all`로 변경되었다. 초기 kNN smoke 실행을 재현하려면 `--graph-mode knn`을 명시한다.

현재 코드의 추가 검증 결과:

- 자동 테스트 **16 passed**. 전체 같은-tissue cell 연결, learned p의 역거리 수식 및 gradient, sender gradient checkpointing 동등성, cached/uncached validation 동등성, all+learned 학습/checkpoint round trip 검사를 추가했다.
- `python scripts/prepare_fish.py` 완료: 저장소에 포함한 FISH h5ad/coordinates 두 파일의 SHA-256, shape, metadata 검증 통과. 7,496 cells × 500 genes 및 eligible receiver 3,452개 확인.
- README의 bash code block 4개를 `bash -n`으로 문법 검사했다. 시스템 패키지 설치·GPU 패키지 설치·GitHub clone을 새 서버에서 실제 수행한 것은 아니다. 문서의 experiment2 GitHub 주소는 공개 비인증 접근에서 404였으므로 코드와 `data/`의 게시가 선행되어야 한다.
- 실제 FISH 전체 **7,496개 cell**을 대상으로 `all + learned`의 reconstruction forward/backward 및 optimizer 1 step을 CPU에서 실행했다. receiver 2개, **receiver당 sender 7,495개**, hidden=8, sender chunk=32, gradient checkpointing 사용. 원래 train/validation split과 학습 중 validation sender 완전 masking을 유지했다.
- 해당 step의 masked Huber=0.06518279, raw-p gradient=-9.84619e-7, p는 **1.0 → 1.00062251**로 갱신되었다. auxiliary 없이 reconstruction에서 거리 parameter까지 gradient가 연결됨을 확인했다.

이 추가 실데이터 검증은 전체 epoch가 아닌 1 step이며, 모델 성능이나 생물학적 성공을 의미하지 않는다. all mode의 50 epoch GPU 학습은 실행하지 않았다.
