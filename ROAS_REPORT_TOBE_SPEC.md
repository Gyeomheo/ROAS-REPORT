# Weekly ROAS Report To-Be Spec (v0.2)

Last Updated: 2026-04-17
Scope: `C:\Weekly ROAS\marketing_orchestrator`

---

## 1) 목적

GMPD RAW 데이터를 입력받아, LMDI 분해 → 법인별 이슈파인딩 → 원인분석(RCA) → 액션아이템을 자동 산출한다.

산출물 목표:

1. **엑셀 리포트**: `ROAS Report Format.xlsx` 템플릿에 법인별 Issue Finding / Action Item 자동 채움
2. **기계용 JSON**: 후단 RAG/자동화가 읽을 수 있는 fact-centered 정본
3. **사람용 HTML**: 주간 회의용 요약 보고서

핵심 원칙:

1. KPI/KPD 시트는 범위 밖 — GMPD RAW 기반 LMDI 분석에만 집중
2. 계산(impact/root-cause)은 결정론 로직 유지, 사람용 표현은 별도
3. 법인별 결과는 빈 상태 없이 최소 1개 토픽 보장
4. JSON이 단일 진실 원천(single source of truth) — HTML/Excel 모두 JSON 기준으로 렌더

---

## 2) 현재 상태 (AS-IS) — 코드 검증 기반

### 2.1 파이프라인 흐름

```
GMPD RAW Excel
  → ingestion.py (Polars DataFrame)
  → impact.py (ImpactEngine: LMDI 분해, drill-down, issue/improve 선정)
  → root_cause.py (RootCauseEngine: CVR/AOV/CPC driver 판별)
  → recommendation.py (rule-based 액션텍스트 생성)
  → report_service.py (top3 선정, subsidiary_reports 조립, JSON/HTML/Excel 저장)
```

### 2.2 확인된 문제점

**① JSON/HTML 토픽 불일치**
- `html_report_legacy.py`에 독립적인 `_fallback_topics_for_sub()` 함수 존재 (434-502행)
- `_merge_topics()` (505-564행)가 JSON의 top3 + 자체 생성 fallback을 합산
- 기여율 계산 방식도 다름: JSON은 impact_contribution_pct 기반, HTML fallback은 revenue_delta 기반
- **결과**: JSON에는 빈 top3인데 HTML에는 3개 토픽이 표시되는 불일치 발생

**② 다양성 제약의 부작용**
- `selectors.py`의 `top3_by_subsidiary()` (22-122행)에 strict/semi/pair/fill 4단계 diversity 로직
- 의도: 동일 CHANNEL/DIVISION이 3개 슬롯 독점 방지 (예: MX Paid Search가 3개 다 차지)
- 부작용: 실제 최대 영향 토픽이 diversity 필터에 걸려 2-3위로 밀리거나 탈락
- 1위는 가장 큰 영향을 가진 항목이므로 diversity 제한 없이 통과시켜야 함

**③ Null Product 미필터**
- `path.PRODUCT`가 빈 값/NULL/N/A인 토픽이 top3에 포함될 수 있음
- 현재 코드에 필터 로직 없음

**④ Action 코드 미구조화**
- `recommendation.py`가 mode_tag × driver × channel_type으로 분기하는 ~18개 하드코딩 한글 텍스트
- 코드 식별자(action_code) 없이 문장만 저장 → RAG/자동화 활용 어려움
- checklist도 "1) ...; 2) ...; 3) ..." 단일 문자열로 반환

**⑤ 엑셀 템플릿 출력 부재**
- 현재 `summary.xlsx`는 분석 raw dump이지, Artience 리포트 형식의 Issue Finding/Action Item 리포트가 아님
- `ROAS Report Format.xlsx` 템플릿 채움 기능 없음

### 2.3 관련 코드 (현행)

| 파일 | 역할 |
|------|------|
| `src/impact.py` | LMDI 분해 + drill-down, `_select_issues()` / `_select_improvements()` |
| `src/root_cause.py` | CVR/AOV/CPC driver 판별, tag(NEW/LOW_VOL/GONE/UNDEFINED/NORMAL) 부여 |
| `src/domain/recommendation.py` | mode_tag × driver × channel_type → 한글 액션 텍스트 |
| `src/application/report_service.py` | `_to_report_rows()`, `_top3_by_subsidiary()`, `_build_subsidiary_reports()`, JSON/HTML/Excel 저장 |
| `src/application/reporting/selectors.py` | `top3_by_subsidiary()` (diversity dedup), `choose_subsidiary_mode()` |
| `src/infrastructure/html_report_legacy.py` | HTML 렌더 + **독립 fallback 생성** (문제의 원인) |
| `src/infrastructure/report_exporter.py` | `save_summary_json()`, `save_summary_html()` |

---

## 3) To-Be 산출물 구조

### 3.1 정규 산출물

1. `output/summary.json` — 기계용 정본 (단일 진실 원천)
2. `output/summary.html` — 사람용 보고서 (JSON 기준 렌더)
3. `output/summary.xlsx` — 분석 raw dump (현행 유지)
4. **`output/roas_report.xlsx`** — **신규**: ROAS Report Format 템플릿에 채운 결과

### 3.2 roas_report.xlsx 템플릿 매핑

템플릿: `C:\Weekly ROAS\Reff\ROAS Report Format.xlsx`

운영 규칙:
1. 템플릿에 없는 법인(신규/추가)은 시트 생성 없이 스킵
2. `Sheet1` 문구는 법인 `mode(ISSUE/IMPROVE)`에 따라 변경
3. 법인 시트 `B2/B3`는 comparison_meta 기반 텍스트로 덮어씀

**Sheet1 (요약)**:
| 행 | Col B | Col C | Col D | Col E-P (merged) |
|---|---|---|---|---|
| 3 | SEC | 김병주 | 이슈 | ← top3 issue summary 텍스트 |
| 4 | | | 원인 분석 | ← primary_driver + detail 요약 |
| 5 | | | Action Items | ← recommended_actions 요약 |
| 6 | SEAU | 박미선 | 이슈 | ... |
| ... | ... | ... | ... | ... |

**법인별 시트 (SEC, SEAU, SEDA, SIEL, SETK)**:
| 셀 | 내용 |
|---|---|
| B1 | `Performance Marketing Report \| {SUBSIDIARY}` |
| B2-B3 | 데이터 소스 + 기간 (comparison_meta 기반 자동 생성) |
| B5 | ■ Issue Finding |
| C8 (merged C8:P8) | Issue #1 상세 (summary_text + detail_text) |
| C9 (merged C9:P9) | Issue #2 상세 |
| C10 (merged C10:P10) | Issue #3 상세 |
| B12 | ■ Action Item |
| C15 (merged C15:P15) | Action #1 (recommended_actions + action_checklist) |
| C16 (merged C16:P16) | Action #2 |
| C17 (merged C17:P17) | Action #3 |

---

## 4) summary.json 변경사항

기존 키 유지 + 아래 필드 추가. (하위호환)

### 4.1 Top-level 추가

| 필드 | 설명 | 비고 |
|------|------|------|
| `schema_version` | `"2.0.0"` | 신규 |

### 4.2 subsidiary_reports[] 추가 필드

| 필드 | 설명 | 비고 |
|------|------|------|
| `mode_reason` | 모드 결정 사유 코드 | 신규. 예: `roas_delta_negative`, `fallback_mode_switch`, `fallback_aggregation` |
| `selection_meta` | 선정 과정 메타 | 신규. `{candidates_count, filtered_count, fallback_used}` |

### 4.3 top3[] 추가 필드

현행 필드(path, mode_tag, tag, primary_driver, impact_contribution_pct/text, selection_rank/pool_size, topic_roas_curr/prev/delta, summary_text, detail_text, recommended_actions, action_checklist 등)는 유지.

| 필드 | 설명 | 비고 |
|------|------|------|
| `action_code` | 구조화된 액션 코드 | 신규. 아래 5절 참조 |
| `checklist_codes` | 구조화된 체크리스트 코드 배열 | 신규. 아래 5절 참조 |
| `is_fallback` | fallback 토픽 여부 | 신규. boolean |
| `fallback_basis` | fallback 사유 코드 | 신규. `null` or 사유 코드 |

---

## 5) Action 코드 체계

현재 `recommendation.py`는 3차원 분기: `mode_tag × primary_driver × channel_type`
- channel_type: SEARCH / PMAX / OTHER (recommendation.py의 `channel_strategy_type()` 기준)

### 5.1 action_code (18개)

특수 태그 (mode_tag 무관):
| 코드 | 조건 |
|------|------|
| `TAG_LOW_VOL_MONITOR` | tag == LOW_VOL |
| `TAG_NEW_STABILIZE` | tag == NEW |
| `TAG_GONE_COVERAGE_CHECK` | tag == GONE |
| `TAG_UNDEFINED_DATA_QC` | tag == UNDEFINED |

ISSUE 모드 (tag == NORMAL):
| 코드 | 조건 |
|------|------|
| `ISSUE_CPC_SEARCH` | driver=CPC, channel=SEARCH |
| `ISSUE_CPC_PMAX` | driver=CPC, channel=PMAX |
| `ISSUE_CPC_OTHER` | driver=CPC, channel=OTHER |
| `ISSUE_CVR_SEARCH` | driver=CVR, channel=SEARCH |
| `ISSUE_CVR_PMAX` | driver=CVR, channel=PMAX |
| `ISSUE_CVR_OTHER` | driver=CVR, channel=OTHER |
| `ISSUE_AOV_SEARCH` | driver=AOV, channel=SEARCH |
| `ISSUE_AOV_PMAX` | driver=AOV, channel=PMAX |
| `ISSUE_AOV_OTHER` | driver=AOV, channel=OTHER |

IMPROVE/DEFENSE 모드 (tag == NORMAL):
| 코드 | 조건 |
|------|------|
| `IMPROVE_CPC_SUSTAIN` | driver=CPC |
| `IMPROVE_CVR_SUSTAIN` | driver=CVR |
| `IMPROVE_AOV_SUSTAIN` | driver=AOV |

driver=NONE:
| 코드 | 조건 |
|------|------|
| `ISSUE_NONE_OBSERVE` | mode=ISSUE, driver=NONE |

### 5.2 checklist_codes

각 action_code에 대응하는 체크리스트 코드 배열. 예:
- `ISSUE_CPC_SEARCH` → `["NEGATIVE_KEYWORD_EXPAND", "BID_CAP_CPC", "QUERY_INTENT_ALIGNMENT"]`
- `ISSUE_CVR_SEARCH` → `["LANDING_MATCH_OPT", "QUERY_INTENT_ALIGNMENT", "AUDIENCE_REQUALIFY"]`

구체적 매핑은 현행 `recommendation.py`의 `action_checklist()` 텍스트에서 1:1 추출하여 정의.

---

## 6) Top3 선정 규칙 (To-Be)

### 6.1 후보 유효성 필터 (신규)

`top3_by_subsidiary()` 진입 전에 다음 조건을 적용:

1. `path.PRODUCT`가 빈 값 / `NULL` / `N/A` / `NONE`이면 **제외**
2. `impact_contribution_pct` 또는 `topic_roas_delta` 또는 driver YoY basis 중 1개 이상 유효 (현행 `_is_actionable()` 유지)
3. `mode_tag`와 Impact_adj 부호가 모순이면 제외 (ISSUE인데 Impact_adj > 0)

### 6.2 스코어 (현행 유지)

`selectors.py`의 `_metric()` 우선순위 (코드 검증 완료, 현행과 동일):

1. `abs(topic_roas_delta)` — 있으면 우선
2. `impact_contribution_pct` — roas_delta 없을 때
3. `abs(Impact_adj)` — 나머지

### 6.3 다양성 제약 — 변경점

**현행**: strict → semi → pair → fill 4단계. 모든 슬롯에 동일 적용.

**To-Be**: **1위는 diversity 제한 없이 무조건 통과**, 2-3위부터 diversity 적용.

이유: 1위는 해당 법인에서 가장 큰 영향을 가진 항목이므로, diversity 필터로 인해 영향력 순위가 왜곡되면 안 됨. MX Paid Search가 실제로 가장 큰 이슈라면 1위에 노출되어야 한다.

변경 방법 (`selectors.py`):
```python
# 1위: 무조건 최고 스코어 항목
if ordered:
    selected.append(ordered[0])
    # 1위의 triplet/channel/bu를 seen에 등록
    ...

# 2-3위: 기존 strict/semi/pair/fill 로직 적용 (ordered[1:]부터)
```

### 6.4 법인별 모드 결정 (현행 유지 + fallback 강화)

현행 (`choose_subsidiary_mode()`):
1. `roas_delta < 0` → ISSUE 모드 (issues 있으면), 없으면 IMPROVE로 전환
2. `roas_delta >= 0` → IMPROVE 모드 (improves 있으면), 없으면 ISSUE로 전환

To-Be 추가:
3. 양쪽 모두 0건이면 → **집계 fallback 토픽 생성** (아래 7절)
4. `mode_reason` 필드에 결정 사유 기록

### 6.5 최소 보장

법인별 `top3`: 최소 1건, 최대 3건 보장.

---

## 7) Fallback 규칙 (빈 top3 방지)

현재 `html_report_legacy.py`에만 있는 fallback 로직을 `report_service.py`로 승격.

### 7.1 승격 대상

`html_report_legacy.py`의 `_fallback_topics_for_sub()` (434-502행) 로직:
- product-level 집계 데이터에서 revenue_delta 기준 상위 토픽 생성
- _driver_from_metrics()로 driver 판별
- 이 로직을 `report_service.py`의 `_build_subsidiary_reports()` 내부로 이동

### 7.2 fallback 순서

1. RCA 기반 토픽 사용 (정상 경로)
2. 부족분은 집계 fallback 토픽으로 보강
3. 보강 토픽에 `is_fallback=true`, `fallback_basis` 기록

fallback_basis 코드:
- `NO_RCA_TOPIC_IN_MODE` — 해당 모드(ISSUE/IMPROVE)에 RCA 토픽 0건
- `INSUFFICIENT_ACTIONABLE_TOPICS` — actionable 토픽 부족
- `NULL_PRODUCT_FILTERED` — null product 필터 후 부족

### 7.3 HTML fallback 제거

fallback 승격 후 `html_report_legacy.py`에서:
- `_fallback_topics_for_sub()` 삭제
- `_merge_topics()` 삭제
- HTML은 `summary.json`의 `subsidiary_reports[].top3`만 렌더 (읽기 전용)

---

## 8) JSON/HTML/Excel 일관성 규칙

단일 진실 원천: `summary.json → subsidiary_reports[].top3`

| 산출물 | 역할 | 토픽 소스 |
|--------|------|-----------|
| summary.json | 정본 | `subsidiary_reports[].top3` |
| summary.html | 사람용 렌더 | JSON의 top3 읽기 전용 |
| roas_report.xlsx | 템플릿 채움 | JSON의 top3 읽기 전용 |
| summary.xlsx | 분석 raw dump | 현행 유지 (별도) |

금지 사항:
- HTML 내 독자적 fallback/재선정
- Excel 렌더 시 JSON과 다른 토픽 선정

---

## 9) 구현 포인트 (파일 단위)

| 순서 | 파일 | 작업 |
|------|------|------|
| 1 | `src/application/reporting/selectors.py` | 1위 diversity 면제, null product 필터 추가 |
| 2 | `src/application/report_service.py` | fallback 로직 승격, mode_reason/selection_meta/action_code 추가 |
| 3 | `src/domain/recommendation.py` | `action_code()` 함수 추가 (기존 텍스트 함수와 병행) |
| 4 | `src/infrastructure/html_report_legacy.py` | fallback 독립 생성 제거, JSON top3 읽기 전용 렌더로 단순화 |
| 5 | `src/infrastructure/report_exporter.py` | `save_roas_report_excel()` 신설 — 템플릿 복사 후 채움 |
| 6 | `tests/test_regressions.py` | 회귀 테스트 확장 |

---

## 10) roas_report.xlsx 생성 로직

```python
def save_roas_report_excel(
    output_path: Path,
    template_path: Path,
    subsidiary_reports: List[Dict],
    comparison_meta: Dict,
) -> None:
    # 1. 템플릿 복사
    # 2. Sheet1 요약: 법인 mode(ISSUE/IMPROVE)에 맞춰 문구/요약 작성, E열(merged E:P) 채움
    # 3. 법인별 시트: B2-B3 텍스트 덮어쓰기, Issue Finding C8-C10, Action Item C15-C17
    # 4. top3 < 3건이면 빈 행은 공란 유지
    # 5. 템플릿에 없는 법인 시트는 스킵
```

Issue Finding 셀 내용 (C8 등):
```
{summary_text}
{detail_text 중 1-3행 요약}
```

Action Item 셀 내용 (C15 등):
```
{recommended_actions}
{action_checklist}
```

Sheet1 이슈 셀 (E3 등):
```
1. {top3[0].summary_text 한줄 요약}
2. {top3[1].summary_text 한줄 요약}
3. {top3[2].summary_text 한줄 요약}
```

---

## 11) 테스트 기준

### 11.1 필수 회귀 테스트

1. `roas_delta < 0` & issue 비어도 법인별 top3 최소 1개 생성
2. top3 중복 규칙: `(CHANNEL, DIVISION, PRODUCT)` 중복 없음
3. 1위는 항상 최고 스코어 항목 (diversity 면제 검증)
4. `NULL/N/A/빈 PRODUCT`는 top3 제외
5. JSON top3와 HTML 렌더 토픽 수 일치
6. `action_code`가 driver/tag/channel_type 조합과 일관
7. fallback 토픽에 `is_fallback=true` 마킹 확인

### 11.2 현행 테스트 커버리지 (참고)

현재 6개 테스트:
- `test_special_tag_preserves_none_driver`
- `test_wide_engine_frame_keeps_zero_current_spend_gone_candidate`
- `test_long_engine_frame_applies_ext_revenue_rules_and_gross_orders`
- `test_insight_text_uses_clean_korean_strings`
- `test_build_run_metadata_includes_hash_and_cutoff_key`
- `test_build_run_metadata_uses_metadata_hash_when_file_open_is_blocked`

미커버 영역: recommendation 로직, top3 선정 로직, fallback 로직, action_code 일관성

---

## 12) 단계별 적용안

### Phase 1 (즉시)

1. 1위 diversity 면제 (`selectors.py`)
2. null product 필터 (`selectors.py`)
3. fallback 승격: `html_report_legacy.py` → `report_service.py`
4. HTML fallback 독립 생성 제거
5. `roas_report.xlsx` 템플릿 채움 기능 신설

### Phase 2

1. `action_code` / `checklist_codes` 구조화
2. `schema_version`, `mode_reason`, `selection_meta` 필드 추가
3. 회귀 테스트 확장

### Phase 3 (선택)

1. `topic_catalog` 기반 RAG query fingerprint
2. 과거 케이스 매칭 자동화 연동
