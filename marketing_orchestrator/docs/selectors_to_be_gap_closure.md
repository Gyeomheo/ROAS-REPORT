# selectors.py TO-BE Gap Closure (2026-04-21)

## 1) 문서 목적
- `selectors_as_is.md`와 W16 감사 결과를 기준으로, gap을 메우는 구현 사양을 정의한다.
- 구현 대상은 Phase D(`selectors.py`)이며, 바로 코드 반영 가능한 수준으로 작성한다.

## 2) 참조 근거
- AS-IS 문서:
  - `C:\Weekly ROAS\marketing_orchestrator\docs\selectors_as_is.md`
- 감사/대조:
  - `C:\Weekly ROAS\Reff\Total_Internal ROAS Report_W16 - Artience.xlsx`
  - `C:\Weekly ROAS\Reff\Total_Internal ROAS Report_W15_Artience.xlsx`
  - `C:\Weekly ROAS\marketing_orchestrator\output\ROAS Report Format_filled.xlsx`
  - `C:\Weekly ROAS\marketing_orchestrator\docs\reff_issue_copy_samples_w15_w16.md`
  - (운영 노트) `C:\Users\Heo Gyeom\Desktop\obsidian-mind\work\active\2026-04-21-roas-format-w16-gap-audit.md`

## 3) 갭 정의와 목표

| gap | AS-IS | TO-BE 목표 | 우선순위 |
|---|---|---|---|
| G1 null/vague product 필터 없음 | `MX CROSS PRODUCTS` 등 generic product가 정상 후보 포함 | valid 후보에서 제외, 필요 시 마지막 fill-null에서만 제한 허용 | P1 |
| G2 top1 diversity 면제 없음 | 영향력 1위가 strict/semi/pair에서 밀릴 수 있음 | top1은 무조건 선선정(면제) | P1 |
| G3 ISSUE threshold 없음 | `roas_delta=-0.001`도 ISSUE 분기 가능 | 임계값 미만 하락만 ISSUE 분기 | P2 |
| G4 fill 단계 product 중복 억제 없음 | 같은 product가 channel만 달라 중복 노출 가능 | fill 단계에서 product 중복 억제 | P1 |

## 4) TO-BE 규칙 (실무 기준 반영)

### 4.1 이슈 선정 단위 전환 (핵심)
- W16 실제 문구 단위는 `DIVISION + CHANNEL` 중심(`MX Paid Search`, `DA Paid Social`)이다.
- TO-BE는 내부 계산 후보를 먼저 `DIVISION+CHANNEL` 관점으로 집계/평가하고, `PRODUCT`는 설명 보조 축으로 강등한다.
- 결과적으로 `MX CROSS PRODUCTS` 같은 generic product가 이슈 타이틀을 점유하지 못하게 한다.

### 4.2 후보 풀 분리
- `ordered`를 아래 두 집합으로 분리:
  - `valid_candidates`: null-like product가 아닌 후보
  - `null_fallback`: null-like product 후보
- null-like product 기본 목록:
  - `MX CROSS PRODUCTS`, `DA CROSS PRODUCTS`, `VD CROSS PRODUCTS`, `CROSS PRODUCTS`, `CROSS DIVISION`, `OTHERS`, `MX OTHERS`, `DA OTHERS`, `VD OTHERS`, 빈값 계열
- null-like는 기본 제외, top3를 채우지 못할 때만 fallback에서 최대 1개 허용.

### 4.3 ISSUE 모드 하드 게이트 (노이즈 차단)
- `choose_subsidiary_mode()`의 단순 `roas_delta < 0` 분기를 폐기하고 게이트 적용:
  - 기본 분기: `roas_delta < ROAS_ISSUE_THRESHOLD`일 때만 ISSUE 후보 평가
  - 기본값: `ROAS_ISSUE_THRESHOLD = -0.03` (환경변수 `ROAS_ISSUE_THRESHOLD`)
- ISSUE 후보 통과 조건(아래 중 1개 이상):
  - `roas_yoy <= -0.15`
  - `revenue_yoy <= -0.20` 그리고 `spend_yoy >= 0`
  - `impact_contribution_pct >= 0.10`
- Volume floor 권장(소표본 노이즈 차단):
  - `max(revenue_curr, revenue_prev) >= 100000` 또는 `max(spend_curr, spend_prev) >= 10000`

### 4.4 이슈 점수 재정의 (실무 피해도 기준)
- TO-BE issue score:
  - `score = w1*roas_drop + w2*revenue_drop + w3*spend_inefficiency + w4*impact_share`
- 기본 권장 가중치:
  - `w1=0.40`, `w2=0.30`, `w3=0.15`, `w4=0.15`
- tie-breaker:
  - `impact_contribution_pct` 높은 순
  - `revenue_delta` 절대 감소 큰 순

### 4.5 top1 선선정 (diversity 면제)
- 정렬 1위는 조건 검사 없이 `selected[0]`에 선입한다.
- strict/semi/pair/fill은 2~3위 슬롯에만 적용한다.

### 4.6 diversity는 후처리로만 적용
- diversity는 핵심 이슈를 제거하지 않는 방향으로 제한:
  - top1 제외 2~3위에서만 strict -> semi -> pair -> fill
- 목적: "다양성 확보"와 "핵심 이슈 보존" 균형.

### 4.7 fill 단계 product 중복 억제
- `used_products` 집합 추가.
- `fill` 패스에서도 이미 선택된 product는 스킵.
- 예외: valid가 부족한 경우 `null_fallback`에서 최대 1개.

## 5) 의사코드 (실무형)

```python
candidates = build_candidates(rows)
dc_groups = aggregate_by_division_channel(candidates)  # 실무 이슈 단위
scored = score_issue_candidates(dc_groups, weights=ISSUE_SCORE_WEIGHTS)

valid_candidates = [x for x in scored if not is_null_like_product(x)]
null_fallback = [x for x in scored if is_null_like_product(x)]

gated = [
    x for x in valid_candidates
    if passes_issue_gate(x, roas_threshold=-0.03, yoy_rules=True, volume_floor=True)
]

selected = []
state = init_state_sets()

if gated:
    pick_unconditionally(gated[0], state, selected)  # top1 immunity

rest = [x for x in gated if x is not selected_top1]
pick_by_stage(rest, stage="strict", selected=selected, state=state, max_n=3)
pick_by_stage(rest, stage="semi", selected=selected, state=state, max_n=3)
pick_by_stage(rest, stage="pair", selected=selected, state=state, max_n=3)
pick_by_stage(rest, stage="fill", selected=selected, state=state, max_n=3, dedupe_product=True)

if len(selected) < 3:
    pick_from_null_fallback(
        null_fallback,
        selected=selected,
        state=state,
        max_n=3,
        dedupe_product=True,
        max_null_items=1,
    )
```

## 6) W16 전 탭 대조 반영 포인트 (SEC 외 포함)
- `SEC`: Paid Search 2건 중심, 하락폭 큼(-48%, -69%), Action 2건.
- `SEAU`: Paid Social 중심 2건, Spend 증가 대비 Revenue 미개선.
- `SEDA`: Paid Search 2건, ROAS/Revenue 하락 이슈 명확.
- `SIEL`: Paid 축소/효율 저하 복합 이슈 2건.
- `SETK`: Paid 양호 + Organic 하락 관찰형 이슈 2건.
- 공통 패턴:
  - 대부분 C8/C9에 핵심 이슈 2건, C15 중심 액션 1~2건.
  - product 이름보다 division/channel 중심 서술.
- current filled의 문제:
  - `SEDA`에 `MX CROSS PRODUCTS` 중복 관측.
  - `SEC/SEAU/SETK` 법인 탭 본문 공란.

## 7) 구현 범위 (파일 단위)
- 필수 변경:
  - `C:\Weekly ROAS\marketing_orchestrator\src\application\reporting\selectors.py`
- 연계 변경(권장):
  - `C:\Weekly ROAS\marketing_orchestrator\src\application\report_service.py` (mode 분기/포맷 연동)
  - 공용 상수 모듈 신설: `src/domain/product_constants.py` (`ingestion/selectors` 공유)
- 테스트:
  - `C:\Weekly ROAS\marketing_orchestrator\tests\test_regressions.py`
  - 신규 케이스:
    - top1 면제
    - null/vague product 제외
    - fill 단계 product dedupe
    - `roas_delta=-0.001` -> IMPROVE
    - W16 샘플 fixture Top2 매칭

## 8) 개발기획 (실행 계획)

### Phase A — 기준선 고정 (0.5일)
- W16/filled 비교 추출 스크립트 고정.
- 법인별 Top2 정답셋(division+channel) 파일화.

### Phase B — selector 코어 리팩터 (1일)
- 후보 분리(valid/null), top1 면제, fill dedupe 구현.
- ISSUE 게이트(임계값/volume floor) 구현.
- issue score 계산 함수 분리.

### Phase C — 보정/튜닝 (0.5일)
- W14~W16 기준 threshold/가중치 튜닝.
- 기본값 확정:
  - `ROAS_ISSUE_THRESHOLD=-0.03`
  - `ROAS_ISSUE_MIN_YOY=-0.15`
  - `REV_ISSUE_MIN_YOY=-0.20`

### Phase D — 회귀/배포 준비 (0.5일)
- 테스트 통과 + 샘플 실행 비교 리포트 생성.
- 실행 로그에 법인별 선택 근거(게이트 통과 사유/점수) 남기기.

## 9) 수용 기준 (Acceptance)
- null-like product가 top3에서 기본 제외됨.
- 동일 product 중복 노출이 fill 단계에서 제어됨.
- top1 영향 토픽이 항상 포함됨.
- ISSUE/IMPROVE 분기가 threshold 정책과 일치함.
- W16 정답셋 기준 법인별 Top2 매칭률 >= 70%.
- 기존 회귀 테스트 + 신규 selector 테스트 통과.

## 10) 결정 필요 항목 (최종 확정용)
1. `ROAS_ISSUE_THRESHOLD` 최종값:
   - 제안: `-0.03`로 시작, W14~W16로 보정
2. volume floor 기준값:
   - 제안: revenue 100k / spend 10k
3. null-like 상수 위치:
   - 제안: 공용 상수 모듈
4. null fallback 허용 개수:
   - 제안: 최대 1개

## 11) 실행 순서 제안 (이번 주)
1. selectors 코어 규칙(B단계) 구현
2. regression + W16 비교 테스트 추가
3. threshold/가중치 튜닝
4. 본 실행(`main.py`) 후 법인별 결과 diff 리뷰
5. 커밋/릴리즈
