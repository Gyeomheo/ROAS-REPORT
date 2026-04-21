# Level 2 Diagnostic Spec — CPC / CVR / AOV 파생 딥다이브

Last Updated: 2026-04-20  
Scope: `C:\Weekly ROAS\marketing_orchestrator`  
선행 문서: `ROAS_REPORT_TOBE_SPEC.md`

---

## 1) 목적

`root_cause.py`가 `primary_driver = CPC / CVR / AOV`를 판별한 뒤,  
그 원인이 **어느 파생 지표에서 왔는지** 한 단계 더 내려가는 Level 2 진단 레이어를 추가한다.

```
[현재]
Spend / Revenue / Clicks / Orders
  → dlog_CPC / dlog_CVR / dlog_AOV
  → primary_driver

[추가]
primary_driver = CPC  →  CPM vs CTR 중 무엇이 주도했나?
primary_driver = CVR  →  Clicks 볼륨 문제인가 / 전환율 자체 문제인가?
primary_driver = AOV  →  Revenue 낙폭인가 / Orders 증가 희석인가?
```

---

## 2) 수학적 분해

### CPC

```
CPC = Spend / Clicks
    = CPM / (CTR × 1000)

  CPM = Spend / Impressions × 1000
  CTR = Clicks / Impressions

→ dlog_CPC = dlog_CPM − dlog_CTR

CPC 상승 원인:
  · CPM ↑  →  노출 단가 상승 (경쟁 강도, 오디언스 포화, 입찰 비효율)
  · CTR ↓  →  클릭률 하락 (소재 피로도, 타겟-소재 의도 불일치)
  · 둘 다  →  복합
```

> Impressions 미제공 시: CPM/CTR 불가 → Spend YoY vs Clicks YoY 증감 격차로 대체 진단

### CVR

```
CVR = Orders / Clicks
→ dlog_CVR = dlog_Orders − dlog_Clicks

CVR 하락 원인:
  · Clicks ↑ & Orders ↓  →  트래픽 증가했는데 전환 안 됨 (유입 품질 희석)
  · Clicks 안정 & Orders ↓  →  랜딩 / 오퍼 / 목표 설정 문제
  · Clicks ↓ & Orders ↓↓  →  볼륨 + 전환율 동시 하락 (예산 or 입찰 위축)
  · CTR ↓ (Impressions 있을 때)  →  의도 정합성 문제가 유입 전 단계부터 시작
```

### AOV

```
AOV = Revenue / Orders
→ dlog_AOV = dlog_Revenue − dlog_Orders

AOV 하락 원인:
  · Revenue 낙폭 > Orders 낙폭  →  저단가 SKU 믹스 상승
  · Orders ↑ & Revenue 소폭 ↑  →  볼륨 희석 (낮은 단가 전환 급증)
  · Revenue ↓ & Orders 비슷    →  가격 / 믹스 직접 하락
```

---

## 3) Sub-driver 분류 코드

### CPC sub_driver

| 코드 | 조건 | Impressions 필요 |
|------|------|:---:|
| `CPM_SURGE` | `dlog_CPM > 0` & `dlog_CPM > abs(dlog_CTR)` | ✅ |
| `CTR_DROP` | `dlog_CTR < 0` & `abs(dlog_CTR) > dlog_CPM` | ✅ |
| `CPC_MIXED` | CPM ↑ & CTR ↓ 모두 유의미 | ✅ |
| `SPEND_SURGE` | Impressions 없고 `dlog_Spend − dlog_Clicks > 0.15` | ❌ |
| `CLICK_DROP` | Impressions 없고 `dlog_Clicks < −0.10` | ❌ |
| `CPC_UNKNOWN` | Impressions 없고 격차 작음 | ❌ |

### CVR sub_driver

| 코드 | 조건 |
|------|------|
| `TRAFFIC_QUALITY_DROP` | `dlog_Clicks > 0.10` & `dlog_Orders < 0` |
| `CONVERSION_FUNNEL_DROP` | `abs(dlog_Clicks) < 0.10` & `dlog_Orders < 0` |
| `VOLUME_AND_RATE_DROP` | `dlog_Clicks < 0` & `dlog_Orders < dlog_Clicks` |
| `TRAFFIC_INTENT_DROP` | Impressions 있고 `dlog_CTR < −0.10` & `dlog_CVR < 0` |

### AOV sub_driver

| 코드 | 조건 |
|------|------|
| `PRODUCT_MIX_SHIFT` | `dlog_Revenue < dlog_Orders` (매출 낙폭 > 주문 낙폭) |
| `ORDER_VOLUME_DILUTION` | `dlog_Orders > 0.05` & `dlog_AOV < 0` (주문 늘었는데 AOV 하락) |
| `REVENUE_COMPRESSION` | `dlog_Revenue < −0.05` & `abs(dlog_Orders) < 0.05` (매출만 줄었음) |

---

## 4) 데이터 변경 사항

### 4.1 ingestion.py — Impressions optional 추가

```python
# 현재
RAW_METRIC_MAP = {
    "PLATFORM_SPEND_USD":   "Spend",
    "PLATFORM_REVENUE_USD": "Revenue",
    "PLATFORM_CLICKS":      "Clicks",
    "GROSS_ORDERS":         "Orders",
}

# 추가 (optional — 없어도 파이프라인 중단 안 함)
OPTIONAL_METRIC_MAP = {
    "PLATFORM_IMPRESSIONS": "Impressions",
}
```

- Impressions 컬럼 없으면 `Impressions_curr = None`, `Impressions_prev = None`
- `impressions_available: false`로 마킹, CPM/CTR 계산 스킵

### 4.2 root_cause.py — 신규 dlog 필드 추가

기존 `RCA_FIELDS`에 추가:

```python
# 항상 계산 가능 (기존 데이터)
"dlog_Clicks"   # log(Clicks_curr / Clicks_prev)
"dlog_Orders"   # log(Orders_curr / Orders_prev)
"dlog_Revenue"  # log(Revenue_curr / Revenue_prev)
"dlog_Spend"    # log(Spend_curr / Spend_prev)

# Impressions 있을 때만
"Impressions_curr"
"Impressions_prev"
"CPM_curr"      # Spend_curr / Impressions_curr × 1000
"CPM_prev"
"CTR_curr"      # Clicks_curr / Impressions_curr
"CTR_prev"
"dlog_CPM"
"dlog_CTR"
```

---

## 5) 신규 모듈: `src/domain/level2_diagnosis.py`

### 5.1 데이터 클래스

```python
@dataclass
class Level2Result:
    primary_driver: str          # CPC / CVR / AOV
    sub_driver: str              # 세부 분류 코드
    impressions_available: bool
    metrics: dict                # 진단에 쓰인 YoY 수치
    diagnosis_text: str          # 한국어 원인 설명
    l2_action_supplement: str    # primary action에 추가할 구체 액션
```

### 5.2 compute 흐름

```python
class Level2DiagnosticEngine:

    def compute(self, signals: CampaignSignals, raw_sums: dict) -> Level2Result:
        driver = signals.primary_driver
        if driver == "CPC":
            return self._diagnose_cpc(signals, raw_sums)
        elif driver == "CVR":
            return self._diagnose_cvr(signals, raw_sums)
        elif driver == "AOV":
            return self._diagnose_aov(signals, raw_sums)
        else:
            return Level2Result(
                primary_driver=driver,
                sub_driver="N/A",
                impressions_available=False,
                metrics={},
                diagnosis_text="",
                l2_action_supplement="",
            )
```

### 5.3 CPC 진단

```python
def _diagnose_cpc(self, signals, raw_sums) -> Level2Result:
    dlog_spend  = raw_sums.get("dlog_Spend")
    dlog_clicks = raw_sums.get("dlog_Clicks")
    dlog_cpm    = raw_sums.get("dlog_CPM")
    dlog_ctr    = raw_sums.get("dlog_CTR")
    imp_avail   = dlog_cpm is not None

    if imp_avail:
        if dlog_cpm > 0 and dlog_ctr < 0:
            sub = "CPC_MIXED"
        elif dlog_cpm > abs(dlog_ctr or 0):
            sub = "CPM_SURGE"
        else:
            sub = "CTR_DROP"
    else:
        diff = (dlog_spend or 0) - (dlog_clicks or 0)
        if diff > 0.15:
            sub = "SPEND_SURGE"
        elif (dlog_clicks or 0) < -0.10:
            sub = "CLICK_DROP"
        else:
            sub = "CPC_UNKNOWN"

    return Level2Result(
        primary_driver="CPC",
        sub_driver=sub,
        impressions_available=imp_avail,
        metrics=self._build_cpc_metrics(raw_sums, imp_avail),
        diagnosis_text=self._cpc_text(sub, raw_sums, imp_avail),
        l2_action_supplement=self._cpc_action(sub),
    )
```

### 5.4 CVR 진단

```python
def _diagnose_cvr(self, signals, raw_sums) -> Level2Result:
    dlog_clicks = raw_sums.get("dlog_Clicks") or 0
    dlog_orders = raw_sums.get("dlog_Orders") or 0
    dlog_ctr    = raw_sums.get("dlog_CTR")
    imp_avail   = dlog_ctr is not None

    if imp_avail and dlog_ctr < -0.10 and signals.dlog_cvr < 0:
        sub = "TRAFFIC_INTENT_DROP"
    elif dlog_clicks > 0.10 and dlog_orders < 0:
        sub = "TRAFFIC_QUALITY_DROP"
    elif abs(dlog_clicks) < 0.10 and dlog_orders < 0:
        sub = "CONVERSION_FUNNEL_DROP"
    elif dlog_clicks < 0 and dlog_orders < dlog_clicks:
        sub = "VOLUME_AND_RATE_DROP"
    else:
        sub = "CVR_UNKNOWN"

    return Level2Result(
        primary_driver="CVR",
        sub_driver=sub,
        impressions_available=imp_avail,
        metrics=self._build_cvr_metrics(raw_sums, imp_avail),
        diagnosis_text=self._cvr_text(sub, raw_sums),
        l2_action_supplement=self._cvr_action(sub, signals),
    )
```

### 5.5 AOV 진단

```python
def _diagnose_aov(self, signals, raw_sums) -> Level2Result:
    dlog_rev    = raw_sums.get("dlog_Revenue") or 0
    dlog_orders = raw_sums.get("dlog_Orders") or 0

    if dlog_orders > 0.05 and signals.dlog_aov < 0:
        sub = "ORDER_VOLUME_DILUTION"
    elif dlog_rev < dlog_orders:
        sub = "PRODUCT_MIX_SHIFT"
    elif dlog_rev < -0.05 and abs(dlog_orders) < 0.05:
        sub = "REVENUE_COMPRESSION"
    else:
        sub = "AOV_UNKNOWN"

    return Level2Result(
        primary_driver="AOV",
        sub_driver=sub,
        impressions_available=False,
        metrics=self._build_aov_metrics(raw_sums),
        diagnosis_text=self._aov_text(sub, raw_sums),
        l2_action_supplement=self._aov_action(sub),
    )
```

---

## 6) recommendation.py 확장 — level2_action_supplement()

기존 `action_checklist(signals)` 유지. 보충 함수 신규 추가:

```python
def level2_action_supplement(signals: CampaignSignals, sub_driver: str) -> str:
    driver = signals.primary_driver
    channel_type = channel_strategy_type(signals.channel)

    if driver == "CPC":
        if sub_driver == "CPM_SURGE":
            return (
                "노출 단가 상승 집중 점검: "
                "1) 타겟 오디언스 포화 여부 확인 및 범위 재조정; "
                "2) 경쟁 강도 낮은 시간대/지면으로 예산 분산; "
                "3) 입찰 상한 or Target CPM 재설정"
            )
        if sub_driver == "CTR_DROP":
            return (
                "소재/타겟 정합성 점검: "
                "1) 소재별 CTR 추이 분리 확인 후 저CTR 소재 비중 축소/교체; "
                "2) 쿼리-소재 의도 재정합; "
                "3) A/B 테스트 슬롯으로 신규 소재 CTR 비교"
            )
        if sub_driver == "CPC_MIXED":
            return (
                "CPM + CTR 복합 이슈: "
                "1) 소재 교체로 CTR 회복 우선; "
                "2) 입찰 상한으로 CPM 방어; "
                "3) 오디언스 세분화로 노출 효율 개선"
            )
        if sub_driver == "SPEND_SURGE":
            return (
                "지출 급증 구간 특정: "
                "1) 캠페인/Ad Group별 지출 비중 분해; "
                "2) 입찰 상한 또는 일예산 캡 점검; "
                "3) 급증 구간 검색어/지면 분석"
            )
        if sub_driver == "CLICK_DROP":
            return (
                "Clicks 감소 원인 특정: "
                "1) 입찰 하락 / 예산 소진 여부 확인; "
                "2) 품질점수 하락 여부 점검; "
                "3) 검색량 자체 감소 여부 확인"
            )

    if driver == "CVR":
        if sub_driver == "TRAFFIC_QUALITY_DROP":
            return (
                "유입 품질 점검: "
                "1) Clicks 증가 구간의 검색어/오디언스 의도 분석; "
                "2) 저의도 유입 제외 및 네거티브 확장; "
                "3) 증가한 트래픽의 전환 기여 vs 미기여 세그먼트 분리"
            )
        if sub_driver == "CONVERSION_FUNNEL_DROP":
            return (
                "랜딩 funnel 점검: "
                "1) LP 로딩속도·메시지 정합·CTA 점검; "
                "2) 전환 액션 설정 / 기여 모델 재확인; "
                "3) 오퍼/가격 경쟁력 점검"
            )
        if sub_driver == "VOLUME_AND_RATE_DROP":
            return (
                "볼륨+효율 동시 위축: "
                "1) 예산/입찰 위축 여부 먼저 확인; "
                "2) 유입 감소 채널 특정 후 재활성화 여부 검토; "
                "3) 타겟 재정의 후 단계 증액"
            )
        if sub_driver == "TRAFFIC_INTENT_DROP":
            return (
                "유입 전 단계 의도 저하: "
                "1) CTR 하락 소재/쿼리 특정; "
                "2) 검색 의도와 소재 메시지 재정합; "
                "3) 오디언스 시그널 재구성"
            )

    if driver == "AOV":
        if sub_driver == "PRODUCT_MIX_SHIFT":
            return (
                "SKU 믹스 점검: "
                "1) DIVISION/PRODUCT별 AOV 추이 분해로 저단가 비중 증가 구간 특정; "
                "2) 고AOV SKU 타겟팅 강화 (피드 라벨, tROAS 상향); "
                "3) 저AOV SKU 입찰 하향 또는 분리"
            )
        if sub_driver == "ORDER_VOLUME_DILUTION":
            return (
                "주문 볼륨 희석 점검: "
                "1) 신규 전환 중 저단가 비중 확대 여부 분석; "
                "2) tROAS 상향 또는 value rule 적용; "
                "3) 할인/프로모 유입 과다 여부 점검"
            )
        if sub_driver == "REVENUE_COMPRESSION":
            return (
                "매출 직접 하락 점검: "
                "1) 고AOV SKU 노출 감소 또는 재고 이슈 확인; "
                "2) 할인 정책 변경 영향 분석; "
                "3) 경쟁 가격 변동 모니터링"
            )

    return ""
```

---

## 7) JSON 출력 스키마 — level2_diagnosis 필드

각 `subsidiary_reports[].top3[]` 항목에 추가:

```json
"level2_diagnosis": {
  "sub_driver": "CTR_DROP",
  "impressions_available": true,
  "metrics": {
    "cpc_curr": 1.23,
    "cpc_prev": 0.98,
    "cpc_yoy_pct": 25.5,
    "spend_curr": 14760,
    "spend_prev": 9604,
    "spend_yoy_pct": 53.7,
    "clicks_curr": 12000,
    "clicks_prev": 9800,
    "clicks_yoy_pct": 22.4,
    "impressions_curr": 235000,
    "impressions_prev": 155600,
    "impressions_yoy_pct": 51.0,
    "cpm_curr": 62.8,
    "cpm_prev": 61.7,
    "cpm_yoy_pct": 1.8,
    "ctr_curr": 0.051,
    "ctr_prev": 0.063,
    "ctr_yoy_pct": -19.0
  },
  "diagnosis_text": "CTR이 19% 하락하며 CPC 상승 주도. CPM은 1.8% 상승에 그쳐 노출 단가보다 클릭률 저하가 원인. 소재 피로도 또는 타겟-소재 정합성 점검 필요.",
  "l2_action_supplement": "소재/타겟 정합성 점검: 1) 소재별 CTR 추이 분리 확인 후 저CTR 소재 비중 축소/교체; 2) 쿼리-소재 의도 재정합; 3) A/B 테스트 슬롯으로 신규 소재 CTR 비교"
}
```

CVR 예시:
```json
"level2_diagnosis": {
  "sub_driver": "TRAFFIC_QUALITY_DROP",
  "impressions_available": false,
  "metrics": {
    "cvr_curr": 0.032,
    "cvr_prev": 0.051,
    "cvr_yoy_pct": -37.3,
    "clicks_curr": 18500,
    "clicks_prev": 11200,
    "clicks_yoy_pct": 65.2,
    "orders_curr": 592,
    "orders_prev": 571,
    "orders_yoy_pct": 3.7
  },
  "diagnosis_text": "Clicks 65% 급증했으나 Orders는 3.7% 증가에 그쳐 CVR 37% 하락. 트래픽 볼륨 확대와 동시에 유입 품질 저하 가능성. 저의도 유입 유입 비중 증가 점검 필요.",
  "l2_action_supplement": "유입 품질 점검: 1) Clicks 증가 구간의 검색어/오디언스 의도 분석; 2) 저의도 유입 제외 및 네거티브 확장; 3) 증가한 트래픽의 전환 기여 vs 미기여 세그먼트 분리"
}
```

AOV 예시:
```json
"level2_diagnosis": {
  "sub_driver": "PRODUCT_MIX_SHIFT",
  "impressions_available": false,
  "metrics": {
    "aov_curr": 72000,
    "aov_prev": 91000,
    "aov_yoy_pct": -20.9,
    "revenue_curr": 42624000,
    "revenue_prev": 51961000,
    "revenue_yoy_pct": -18.0,
    "orders_curr": 592,
    "orders_prev": 571,
    "orders_yoy_pct": 3.7
  },
  "diagnosis_text": "Revenue -18%, Orders +3.7%로 매출 낙폭이 주문 증가와 역방향. 저단가 SKU 유입 비중 상승 또는 고단가 SKU 이탈 가능성.",
  "l2_action_supplement": "SKU 믹스 점검: 1) DIVISION/PRODUCT별 AOV 추이 분해로 저단가 비중 증가 구간 특정; 2) 고AOV SKU 타겟팅 강화; 3) 저AOV SKU 입찰 하향 또는 분리"
}
```

---

## 8) Impressions 없을 때 graceful degradation

| 진단 | Impressions 있음 | Impressions 없음 |
|------|:---:|:---:|
| CPC: CPM / CTR 완전 분해 | ✅ | ❌ → Spend/Clicks 격차 대체 |
| CVR: CTR 기반 의도 진단 | ✅ | ❌ → Clicks/Orders 기반 진단 |
| CVR: 볼륨 vs 전환율 분리 | ✅ | ✅ |
| AOV: Revenue / Orders 분해 | ✅ | ✅ (항상 가능) |

```
JSON: "impressions_available": false → cpm_*, ctr_*, impressions_* 필드 모두 null
```

---

## 9) 구현 순서 (Phase 2 후반)

| 순서 | 파일 | 작업 |
|------|------|------|
| 1 | `ingestion.py` | `PLATFORM_IMPRESSIONS` optional 추가. `Impressions_curr/prev` ENGINE_METRIC_COLUMNS 추가 |
| 2 | `root_cause.py` | `dlog_Clicks`, `dlog_Orders`, `dlog_Revenue`, `dlog_Spend` 추가. Impressions 있을 때 `CPM_curr/prev`, `CTR_curr/prev`, `dlog_CPM`, `dlog_CTR` 추가 |
| 3 | `src/domain/level2_diagnosis.py` | 신규. `Level2DiagnosticEngine` + `Level2Result` 구현 |
| 4 | `src/domain/recommendation.py` | `level2_action_supplement(signals, sub_driver)` 추가 |
| 5 | `src/application/report_service.py` | top3 조립 시 `level2_diagnosis` 필드 추가 |
| 6 | `src/infrastructure/html_report_legacy.py` | Level 2 수치 테이블 + 진단 텍스트 렌더링 추가 |
| 7 | `src/infrastructure/report_exporter.py` | `roas_report.xlsx` 법인별 시트에 L2 진단 텍스트 추가 |
| 8 | `tests/test_regressions.py` | sub_driver 분류 일관성 테스트 추가 |

---

## 10) 회귀 테스트 기준 (Level 2)

추가할 테스트:

1. `CPC + Impressions 있을 때` → sub_driver가 CPM_SURGE / CTR_DROP / CPC_MIXED 중 1개
2. `CPC + Impressions 없을 때` → sub_driver가 SPEND_SURGE / CLICK_DROP / CPC_UNKNOWN 중 1개
3. `CVR + Clicks 급증 + Orders 감소` → sub_driver = TRAFFIC_QUALITY_DROP
4. `CVR + Clicks 안정 + Orders 감소` → sub_driver = CONVERSION_FUNNEL_DROP
5. `AOV + Orders 증가 + AOV 하락` → sub_driver = ORDER_VOLUME_DILUTION
6. `AOV + Revenue 낙폭 > Orders 낙폭` → sub_driver = PRODUCT_MIX_SHIFT
7. `level2_diagnosis.metrics`의 yoy_pct 계산 일관성
8. `impressions_available = false` 시 CPM/CTR 필드 null 보장

---

## 11) 의사결정 로그

| 날짜 | 결정 | 이유 |
|------|------|------|
| 2026-04-20 | Level 2를 `level2_diagnosis.py` 독립 모듈로 분리 | 기존 root_cause.py 복잡도 유지. 파이프라인에 영향 없이 붙일 수 있음 |
| 2026-04-20 | Impressions = optional | GMPD RAW에 항상 존재하지 않음. 없어도 AOV/CVR 진단은 가능 |
| 2026-04-20 | sub_driver 분류 threshold 0.10 / 0.15 | dlog 스케일 기준 유의미한 변화 구분. 실데이터 검증 후 조정 가능 |
| 2026-04-20 | l2_action_supplement는 기존 action_checklist에 추가하는 방식 | 기존 액션 체계 교체 없이 더 구체적인 진단을 레이어로 쌓음 |
