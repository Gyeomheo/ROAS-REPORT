# selectors.py AS-IS (2026-04-21)

## 1) 문서 목적
- 현재 `selectors.py`의 실제 동작 로직을 코드 기준으로 정리한다.
- 후속 TO-BE 설계/구현 시 기준선(baseline)으로 사용한다.

## 2) 분석 범위
- 코드:
  - `C:\Weekly ROAS\marketing_orchestrator\src\application\reporting\selectors.py`
  - `C:\Weekly ROAS\marketing_orchestrator\src\application\report_service.py`
- 대조 파일:
  - `C:\Weekly ROAS\Reff\Total_Internal ROAS Report_W16 - Artience.xlsx`
  - `C:\Weekly ROAS\marketing_orchestrator\output\ROAS Report Format_filled.xlsx`

## 3) 실행 컨텍스트
- `report_service.py`에서 selectors 함수 호출:
  - `_top3_by_subsidiary()` -> `reporting_selectors.top3_by_subsidiary(...)`
  - `_choose_subsidiary_mode()` -> `reporting_selectors.choose_subsidiary_mode(...)`
- 현재 호출 파라미터:
  - `MIN_TOPIC_ROAS_ABS_DELTA = 0.03`
  - issues/improves 모두 `descending=True`로 정렬 호출됨

## 4) 함수별 AS-IS

### 4.1 `_topic_roas_delta(item)`
- 우선 `item["topic_roas_delta"]`가 있으면 그 값을 float 변환 후 사용.
- 없으면 `ROAS_curr - ROAS_prev` 계산.
- 둘 다 없으면 `None`.

### 4.2 `top3_by_subsidiary(rows, descending, min_topic_roas_abs_delta=0.03)`

#### 입력/출력
- 입력: 토픽 후보 row 리스트(딕셔너리), 정렬방향 플래그, 최소 ROAS delta 임계값.
- 출력: `{subsidiary: [top3 rows]}`.

#### 내부 처리 순서
1. `diagnosis_subsidiary` 기준 그룹핑.
2. 후보 필터 `_is_actionable(item)`:
   - `topic_roas_delta`가 `None`이면 `YoY basis(CVR/AOV/CPC)`가 있어야 통과.
   - `topic_roas_delta`가 있으면:
     - `YoY basis`가 있거나
     - `abs(topic_roas_delta) >= 0.03`이면 통과.
3. 정렬:
   - 메인 metric `_metric(item)`:
     - 1순위: `abs(topic_roas_delta)` (존재 시)
     - 2순위: `impact_contribution_pct`
     - 3순위: `abs(impact_adj or Impact_adj)`
   - sort key:
     - `descending=True`면 `-_metric(row)`로 내림차순 효과
     - tie-breaker: `(CHANNEL, DIVISION, PRODUCT)` 문자열 오름차순
4. 다양성 기반 선발 (`selected` 최대 3개):
   - 상태 집합:
     - `seen_triplets = {(channel, division, product)}`
     - `used_channel_bu = {(channel, division)}`
     - `used_channels = {channel}`
     - `used_bus = {division}`
   - 4패스 순차 실행:
     - `strict`: 신규 pair + 신규 channel + 신규 division
     - `semi`: 신규 pair + (신규 channel 또는 신규 division)
     - `pair`: 신규 pair
     - `fill`: 제약 없음(단, 동일 triplet 중복은 금지)
5. `selected[:3]` 반환.

#### AS-IS 특성
- top1(영향력 1위) 면제 규칙 없음: diversity 조건에 따라 후순위 밀릴 수 있음.
- product null/vague 제외 규칙 없음: generic product도 정상 후보로 포함됨.
- fill 단계는 product 중복 억제 없음: channel만 다르면 같은 product 재등장 가능.

### 4.3 `choose_subsidiary_mode(subsidiary, perf, issues_by_sub, improves_by_sub)`
- `roas_delta`를 우선 사용, 없으면 `rev_delta` 사용.
- 분기:
  - `roas_delta_value < 0`:
    - issues 있으면 `ISSUE`
    - 없으면 `IMPROVE`
  - `roas_delta_value >= 0`:
    - improves 있으면 `IMPROVE`
    - 없으면 issues 있으면 `ISSUE`
    - 둘 다 없으면 `IMPROVE`
- 별도 threshold 없음(미세한 음수도 ISSUE 분기 가능).

## 5) 실파일 대조 근거 (W16 vs current filled)

### 5.1 시트 구조
- `ROAS Report Format_filled.xlsx`: 6시트 (`Sheet1`, `SEC`, `SEAU`, `SEDA`, `SIEL`, `SETK`)
- `Total_Internal ROAS Report_W16 - Artience.xlsx`: 22시트 (`Sheet1` 없음)

### 5.2 현재 filled에서 selectors 결과가 보이는 패턴
- `SEDA` 탭 `C8/C9`:
  - `SEARCH/MX/MX CROSS PRODUCTS ...`
  - `PMAX/MX/MX CROSS PRODUCTS ...`
- 동일 product(`MX CROSS PRODUCTS`)가 channel만 달라 2건 노출됨.
- 이는 AS-IS 로직에서 triplet 중복만 막고 product 중복을 막지 않는 구조와 일치.

### 5.3 W16 실무 형식
- W16 `SEC` 탭 `C8/C9`:
  - `MX Paid Search ... (-48%) ...`
  - `VA Paid Search ... (-69%) ...`
- 실무 문구는 issue 중심 자연어 요약이며, generic product 반복 노출 패턴이 아님.

## 6) AS-IS 결론
- 현재 `selectors.py`는 "행동 가능 후보 필터 + diversity 기반 top3"라는 기본 구조는 명확하다.
- 다만 실무 리포트 품질 관점에서 아래 4가지 갭이 구조적으로 남아 있다.
  - null/vague product 필터 부재
  - top1 diversity 면제 부재
  - ISSUE threshold 부재
  - fill 단계 product 중복 억제 부재
