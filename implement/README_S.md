# Advanced Flex Travel: MPE 업데이트 (Memory, Priority, Evaluation)

여행 계획 에이전트에 **메모리 재주입(Memory Reinjection)**, **우선순위 랭킹(Priority Ranking)**, 그리고 **자가 평가(Self-Evaluation)** 기능을 도입했습니다. 이를 통해 에이전트는 복잡한 제약조건을 더 잘 기억하고, 충돌이 발생했을 때 우선순위에 따라 현명하게 대처할 수 있게 됩니다.

## 📂 새로운 파일 구조 및 역할

다음과 같은 파일들이 추가되었습니다:

  * **`rank_generator.py`**:
      * 기존 데이터셋에는 우선순위(Priority Rank) 정보가 없기 때문에, 원본 데이터셋을 바탕으로 임의의 랭크 정보를 생성하여 추가해 주는 스크립트입니다.
      * **결과물**: `./evaluation/database_with_ranks` 경로에 새로운 데이터셋을 생성합니다.
  * **`./evaluation/database_with_ranks`**:
      * `rank_generator.py`를 통해 생성된, 우선순위 정보가 포함된 새로운 데이터셋 디렉토리입니다.
  * **`flow_add_mpe.py`**:
      * 새로운 기능들의 핵심 로직이 담긴 파일입니다.
      * 과거의 제약조건을 기억하는 **Memory Reinjection** 기능, **Self Evaluation** 기능과 **Priority Ranks**를 처리하는 함수들이 구현되어 있습니다.
  * **`prompts_add_mpe.py`**:
      * 메모리 블록, 우선순위 가이드라인, **Memory Reinjection, Self-Evaluation, Priority Ranks** 프롬프트 템플릿들을 관리하는 파일입니다.
  * **`evaluate_add_mpe.py`**:
      * 새로운 기능들을 터미널에서 제어할 수 있도록 Config 설정이 추가된 메인 평가 스크립트입니다.
      * `flow_add_mpe`와 `prompts_add_mpe`를 사용하여 평가를 수행합니다.

-----

## 사용 방법 (Usage)

`evaluate_add_mpe.py`를 사용하여 평가를 실행할 수 있습니다. 터미널 Flag를 통해 Baseline부터 세기능이 모두 활성화된 모드까지 자유롭게 설정 가능합니다.

### 기본 명령어 구조

```bash
python evaluate_add_mpe.py \
  --mode [실행모드] \
  --constraints "[제약조건리스트]" \
  --dataset_dir [데이터셋경로] \
  --output_dir [결과저장경로] \
  [기능 활성화 플래그]
```

### 🔧 설정 옵션 가이드

평가 파이프라인을 커스터마이징하기 위한 상세 옵션입니다.

#### 1\. 기본 설정 (Standard Arguments)

기존 Baseline과 동일하게 적용되는 옵션들입니다.

| 인자 | 설명 | 예시 |
| :--- | :--- | :--- |
| `--mode` | 평가 모드 설정 | `--mode single_constraint` |
| `--constraints` | 평가할 제약조건 목록 (쉼표로 구분) | `--constraints "budget,room type"` |
| `--dataset_dir` | 데이터셋 경로 (랭크가 포함된 폴더 사용) | `--dataset_dir "./agents/evaluation/database"` |
| `--output_dir` | 로그 및 결과가 저장될 경로 | `--output_dir "results/single_turn_baseline"` |

#### 2\. MPE 기능 설정 (Feature Flags)

**🧠 메모리 재주입 (Memory Reinjection)**
이전 턴의 제약조건을 기억하여 다시 주입할지 여부를 결정합니다.

  * `--use-memory`: 메모리 기능 사용
  * `--no-memory`: 메모리 기능 미사용

**⭐ 우선순위 랭크 (Priority Ranks)**
제약조건의 중요도(Rank/Weight)를 반영할지 여부를 결정합니다.

  * `--use-priority`: 우선순위 기능 사용
  * `--no-priority`: 우선순위 기능 미사용

**⚖️ 우선순위 타입 (Priority Types)**
우선순위 기능을 켤 때(`--use-priority`), 어떤 방식을 사용할지 지정합니다. (`--priority-type` 옵션 사용)

  * `numerical`: 단순 숫자 가중치 사용 (예: 0.1 \~ 1.0)
  * `label`: 텍스트 라벨 사용 (CRITICAL, HIGH, MEDIUM, LOW)
  * `rank_only`: 단순 순위 숫자 사용 (Rank 1, Rank 2...)
  * `hybrid_rank`: 라벨과 순위를 함께 사용 (예: [HIGH] Rank 3)
  * `hybrid_weight`: 라벨과 가중치를 함께 사용 (예: [HIGH] Weight 0.8)

**✅ 자가 평가 (Self-Evaluation)**
에이전트가 답변을 내기 전에 스스로 계획을 검토하고 수정할지 여부를 결정합니다.

  * `--use-self-eval`: 자가 평가 루프 활성화
  * `--no-self-eval`: 자가 평가 비활성화

-----

## 💻 실행 예시 (Examples)

### 예시 1: 모든 기능 활성화 (Hybrid Weight 모드)

memory reinjection, self evaluation, priority ranks (Hybrid Weight 방식)를 모두 켜고 실행하는 예시입니다.

```bash
python evaluate_add_mpe.py \
  --mode single_constraint \
  --constraints "budget,room type,cuisine,people_number,house_rule" \
  --dataset_dir "./agents/evaluation/database_with_ranks" \
  --output_dir "results/single_turn_full_features" \
  --use-memory \
  --use-priority \
  --priority-type "hybrid_weight" \
  --use-self-eval
```

### 예시 2: 우선순위만 사용 (Label 모드)

메모리와 자가 평가는 끄고, 우선순위 기능만 텍스트 라벨(CRITICAL/LOW 등) 방식으로 실행하는 예시입니다.

```bash
python evaluate_add_mpe.py \
  --mode single_constraint \
  --constraints "budget,room type" \
  --dataset_dir "./agents/evaluation/database_with_ranks" \
  --output_dir "results/single_priority_label_test" \
  --no-memory \
  --use-priority \
  --priority-type "label" \
  --no-self-eval
```

### 예시 3: Baseline (기본 모드)

새로 추가된 MPE 기능들을 모두 끄고, 기존 Baseline과 동일하게 실행하는 예시입니다.

```bash
python evaluate_add_mpe.py \
  --mode single_constraint \
  --constraints "budget,room type" \
  --dataset_dir "./agents/evaluation/database" \
  --output_dir "results/single_baseline" \
  --no-memory \
  --no-priority \
  --no-self-eval
```