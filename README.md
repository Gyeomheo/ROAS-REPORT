# ROAS Report Orchestrator

이 도구는 **작년 같은 기간 대비 매출이 왜 변했는지**를 자동으로 설명해주는 주간 마케팅 리포트 생성기입니다.  
핵심 목적은 마케터가 매체/사업부/상품 단위 원인을 빠르게 찾고, 바로 실행할 액션까지 결정하게 돕는 것입니다.

## 마케터 관점에서 이 도구가 하는 일

1. 성과 Excel을 읽어 분석 가능한 형태로 자동 정리합니다.
2. 매출 변화에 영향이 큰 이슈/개선 Top 항목을 우선순위로 뽑습니다.
3. 변화 원인을 `CPC`, `CVR`, `AOV` 중 무엇이 주도했는지 알려줍니다.
4. 항목별 권장 액션과 체크리스트를 같이 제시합니다.
5. 공유 가능한 결과물(JSON/HTML/Excel)을 한 번에 생성합니다.

## 왜 필요한가

- 주간 보고서 작성에 드는 수작업 시간을 줄입니다.
- "무슨 일이 있었는지(What)" + "왜 그랬는지(So what)" + "다음에 뭘 할지(Now what)"를 한 번에 정리합니다.

## 1) 시스템 아키텍처

```text
main.py (CLI Entrypoint)
  -> application.report_service.run_reporting_pipeline (Orchestrator)
      -> infrastructure.excel_repository.load_input_frame
          -> ingestion.read_input_excel (Polars-first, openpyxl fallback, DQ check)
          -> output/.cache parquet/meta (입력 캐시)
      -> application.analysis_service.run_subsidiary_analysis
          -> ImpactEngine (LMDI 기반 매출 기여 분해 + residual 잔차 보정 + drill-down)
          -> RootCauseEngine (dlog(=delta-log) 기반 driver 추정)
          -> campaign_action_service (도메인 액션 문구/체크리스트)
      -> application.reporting.*
          -> metrics (공통 수치/포맷)
          -> lmdi (division 병렬 기여도)
          -> selectors (Top3 선택 전략)
          -> rendering (서술형 문구)
      -> infrastructure.report_exporter
          -> summary.json / summary.html / summary.xlsx 저장
```

## 2) 설계 철학

### A. Orchestrator-First
- `main.py`는 입력 인자 처리 후 오케스트레이터(`run_reporting_pipeline`)만 호출합니다.
- 파이프라인 제어는 `report_service`에서 단일 책임으로 수행합니다.

### B. Layered Architecture
- `application`: 유스케이스 조합/흐름 제어
- `domain`: 추천 정책/규칙(비즈니스 의사결정)
- `infrastructure`: 입출력, 캐시, 리포트 저장
- 엔진(`impact.py`, `root_cause.py`)은 계산 로직에 집중합니다.

### C. Explainable Math
- 기여도 계산은 LMDI(log-mean) 우선, 불가능한 경우 delta fallback을 사용합니다.
- 부모-자식 합 불일치 잔차(residual)는 1개 항목 몰아주기 대신 `|Impact_raw|` 비례 배분을 사용합니다.

### D. Defensive Data Quality
- 숫자 파싱 실패 비율을 컬럼별로 검사하고, 임계치 초과 시 즉시 실패(Fail-fast)합니다.
- 운영 중 조용한 오염(silent failure)을 줄이는 것을 우선합니다.

### E. Operational Pragmatism
- 입력 전처리 결과를 `output/.cache`에 캐시하여 재실행 시간을 단축합니다.
- Excel I/O는 Polars 우선, 실패 시 openpyxl fallback으로 안정성을 확보합니다.

## 3) 주요 알고리즘 요약 (마케터 버전)

### 먼저 한 줄로
- `ImpactEngine`: 무엇이 매출 증감에 얼마나 영향을 줬는지 찾는 단계
- `RootCauseEngine`: 그 변화가 `CPC/CVR/AOV` 중 무엇 때문인지 찾는 단계

### ImpactEngine (`src/impact.py`)
- 분석 순서: `SUBSIDIARY -> CHANNEL -> DIVISION -> PRODUCT`

`LMDI` 상세:

1. 기본 아이디어  
   부모 매출 변화액(`Parent_revenue_delta`)을 자식 항목(예: division, product)별 기여도로 분해합니다.

2. 로그평균(Log-Mean)  
   `L(x, y) = (x - y) / (ln x - ln y)`  
   단, `x == y`면 `L(x, y) = x`로 처리합니다.

3. 1차 기여도 `Impact_raw` 계산  
   LMDI 유효 조건(부모/자식 현재·전년 매출 모두 양수 + 로그 계산 가능)을 만족하면:

   ```text
   Impact_raw_i
   = ( L(Revenue_curr_i, Revenue_prev_i) / L(Parent_curr, Parent_prev) )
     * ln(Parent_curr / Parent_prev)
     * (Parent_curr - Parent_prev)
   ```

   유효 조건을 만족하지 못하면 안전하게 direct delta fallback:

   ```text
   Impact_raw_i = Revenue_curr_i - Revenue_prev_i
   ```

4. residual(잔차) 보정  
   - `Residual_parent = Parent_delta - Σ Impact_raw_i`
   - 잔차는 한 항목 몰아주기 대신 `|Impact_raw|` 비례 배분:

   ```text
   residual_alloc_i = (|Impact_raw_i| / Σ|Impact_raw|) * Residual_parent
   Impact_adj_i = Impact_raw_i + residual_alloc_i
   ```

   - `Σ|Impact_raw| == 0`이면 자식 수로 균등 배분합니다.

5. 해석용 비율  
   - `Share_childsum = |Impact_adj_i| / Σ|Impact_adj|`
   - `Share_parentdelta = |Impact_adj_i| / |Parent_delta|`

6. drill-down 규칙  
   - `SINGLE`: `top1_share >= 0.45` AND `(top1_share - top2_share) >= 0.15` AND `|top1_impact| >= impact_floor`
   - 그 외:
     - 부모 변화가 음수면 `WIDE_NEG` (음수 기여 상위 최대 3개)
     - 부모 변화가 양수면 `WIDE_POS` (양수 기여 상위 최대 3개)
     - 부모 변화가 0이면 `STOP_ZERO_DELTA`
   - `impact_floor = max(abs_floor_amount, parent_prev_revenue * rel_floor)`

### RootCauseEngine (`src/root_cause.py`)
- 기본 관계: `ROAS = CVR × AOV ÷ CPC`
- 왜 이렇게 되나요? (정의에서 바로 나옵니다)
```text
ROAS = Revenue / Spend
CVR = Orders / Clicks, AOV = Revenue / Orders, CPC = Spend / Clicks
CVR × AOV ÷ CPC
= (Orders/Clicks) × (Revenue/Orders) × (Clicks/Spend)
= Revenue / Spend   (Orders, Clicks가 약분)
```
- `dlog`는 전년 대비 변화를 "방향(+/-)과 강도"로 비교하기 위한 점수입니다.
해석 기준(간단):
- `dlog_CVR`가 크고 +면 CVR 개선 영향이 큼
- `dlog_AOV`가 크고 +면 객단가 개선 영향이 큼
- `dlog_CPC`가 크고 +면 CPC 악화(비용 상승) 영향이 큼
- `primary_driver`는 위 3개 중 실제 결과(악화/개선)와 가장 강하게 맞는 1개를 고른 값입니다.
- 표본이 너무 적거나 해석이 불안정하면 `primary_driver`를 `NONE`으로 둡니다.
- 태그(`tag`)는 `NORMAL`, `LOW_VOL`, `NEW`, `GONE`, `UNDEFINED`를 사용합니다.
  - `LOW_VOL`: `Clicks_curr_sum < 100` 또는 `Orders_curr_sum < 5`
  - `NEW`: 전년 매출 0, 올해 매출 > 0
  - `GONE`: 전년 매출 > 0, 올해 매출 0
  - `UNDEFINED`: 로그분해 불가 케이스

### 용어만 빠르게
- `LMDI`: 총 변화액을 항목별로 분해하는 방법
- `residual`: 분해 후 남는 자투리 오차
- `dlog`: 전년 대비 변화의 방향과 강도를 함께 보는 점수

### 리포트에서 이렇게 보시면 됩니다
- `Impact_adj < 0`: 매출 하락에 기여(이슈 후보)
- `Impact_adj > 0`: 매출 상승에 기여(개선 후보)
- `primary_driver = CPC/CVR/AOV`: 액션 우선순위를 둘 핵심 원인
- `tag = LOW_VOL/UNDEFINED`: 해석 신뢰도 주의

### Reporting Split (`src/application/reporting/`)
- `metrics.py`: 공통 수치/포맷
- `lmdi.py`: division 병렬 기여도(Impact_raw_method, residual 보정, decline/growth 기여율 산출)
- `selectors.py`: Top3 추출 전략
- `rendering.py`: 최종 코멘트 텍스트

## 4) 현재 디렉터리 구조

```text
marketing_orchestrator/
  main.py
  src/
    __init__.py
    ingestion.py
    impact.py
    root_cause.py
    application/
      analysis_service.py
      campaign_action_service.py
      report_service.py
      reporting/
        metrics.py
        lmdi.py
        selectors.py
        rendering.py
    domain/
      models.py
      recommendation.py
    infrastructure/
      excel_repository.py
      report_exporter.py
      html_report_legacy.py
```

## 5) 실행 플로우 (현재 코드 기준)

1. `data/raw/input.xlsx` 로드
2. 스키마 정규화(wide/long 자동 감지), OBJECTIVE/DIVISION 필터, MTD 정렬
3. `SUBSIDIARY` 단위로 ImpactEngine 실행
4. Top campaign 대상 RootCauseEngine 실행
5. 도메인 액션 추천(`recommended_actions`, `action_checklist`) 부여
6. 제품/사업부/법인 레벨 리포트 행 구성 및 Top3 선택
7. `summary.json`, `summary.html`, `summary.xlsx` 생성
8. 단계별 실행 시간 로그 출력

연도 결정 규칙(코드 기준):
- 우선순위: CLI 인자(`--curr-year`, `--prev-year`) > 환경변수(`ROAS_CURRENT_YEAR`, `ROAS_PREVIOUS_YEAR`) > 시스템 현재연도/전년
- 한쪽 연도만 지정되면 다른 한쪽은 자동 보정 (`curr`만 있으면 `prev=curr-1`, `prev`만 있으면 `curr=prev+1`)

## 6) 데이터 계약 (Input Contract)

### A. 공통 차원 컬럼
- `SUBSIDIARY`, `CHANNEL`, `DIVISION`, `PRODUCT`

### B. Wide 포맷 (엔진 입력 형태)
- `Spend_curr`, `Spend_prev`, `Revenue_curr`, `Revenue_prev`
- `Clicks_curr`, `Clicks_prev`, `Orders_curr`, `Orders_prev`

### C. Long 포맷 (원본 집계형)
- 지표 원본 컬럼: `PLATFORM_SPEND_USD`, `GROSS_REVENUE`, `PLATFORM_CLICKS`, `GROSS_ORDERS`
- 기간 컬럼: `Year/YEAR` (+ 선택 `Month/MONTH`, `Day/DAY`)
- 선택 필터 컬럼: `OBJECTIVE` (대소문자 변형 허용)

### D. 자동 필터 정책
- OBJECTIVE: `CONVERSION`만 사용
- DIVISION: `MX`, `VD`, `DA`만 사용
- 연도: 비교 2개 연도(`curr_year`, `prev_year`)
- MTD: 현재연도 최대 월의 1일~최대일 기준으로, 과거연도도 동일 월/일 컷오프 정렬

## 7) 실행 방법

### 사전 준비
- Python 3.10+
- 권장 패키지: `polars`, `openpyxl`

### 입력 파일 위치
- `marketing_orchestrator/data/raw/input.xlsx`

### 실행
```bash
cd marketing_orchestrator
python main.py
```

### 연도 지정 실행
```bash
python main.py --curr-year 2026 --prev-year 2025
```

## 8) 설정값 (Environment Variables)

- `ROAS_CURRENT_YEAR`: `--curr-year` 미지정 시 현재연도 오버라이드
- `ROAS_PREVIOUS_YEAR`: `--prev-year` 미지정 시 전년 오버라이드
- `ROAS_PARSE_ERROR_THRESHOLD`: 숫자 파싱 실패 허용 비율 (기본 `0.01`)
- `ROAS_PARSE_ERROR_THRESHOLD`는 `0~1` 범위만 허용 (범위 밖이면 즉시 예외)

연도 보정 규칙:
- `ROAS_CURRENT_YEAR`만 주면 `prev = curr - 1`
- `ROAS_PREVIOUS_YEAR`만 주면 `curr = prev + 1`

예시:
```bash
set ROAS_PARSE_ERROR_THRESHOLD=0.02
python main.py
```

## 9) 산출물 (Output)

- `marketing_orchestrator/output/summary.json`
  - 비교 메타(`comparison_meta`), 분해 방법(`decomposition_method`), division 병렬기여도(`division_parallel_contribution`), 법인별 리포트, 이슈/개선 Top3 포함
- `marketing_orchestrator/output/summary.html`
  - 요약용 HTML 리포트
- `marketing_orchestrator/output/summary.xlsx`
  - 시트: `issues`, `improvements`, `issues_view`, `improvements_view`, `division_parallel`, `trace`
- `marketing_orchestrator/output/.cache/*`
  - 입력 정규화 결과 캐시(parquet + meta)

참고: `data/`, `output/`은 `.gitignore`에 의해 기본적으로 버전관리 제외됩니다.

## 10) 운영/장애 대응

### Excel 저장 실패
- 파일이 열려 있으면 저장이 실패할 수 있습니다.
- 파이프라인은 예외를 흡수하고 경고를 출력합니다.

### 입력 파싱 실패
- 파싱 실패율이 임계치 초과 시 `ValueError`로 중단됩니다.
- `ROAS_PARSE_ERROR_THRESHOLD`를 환경에 맞게 조정하세요.

### 시트/헤더 불일치
- 우선 `raw` 시트를 시도하고, 실패 시 첫 시트로 fallback합니다.
- 헤더 중복/공백은 ingestion 단계에서 정규화합니다.

## 11) Python API 사용 예시

```python
from src.application.report_service import run_reporting_pipeline

run_reporting_pipeline(curr_year=2026, prev_year=2025)
```

Top-level 패키지에서도 접근 가능합니다:
```python
import src

src.run_reporting_pipeline(curr_year=2026, prev_year=2025)
```

## 12) 현재 인터페이스 상태

- 권장 엔트리포인트: `marketing_orchestrator/main.py`
- 오케스트레이션 API: `from src.application import run_reporting_pipeline`
- 레거시 HTML 렌더러는 `src/infrastructure/html_report_legacy.py`로 이동되어 인프라 계층으로 격리되었습니다.
