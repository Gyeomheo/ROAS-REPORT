"""Domain policies for campaign diagnosis and action recommendations."""

from __future__ import annotations

from src.domain.models import CampaignSignals


def channel_strategy_type(channel: str) -> str:
    channel_text = str(channel or "").strip().upper()
    if "PMAX" in channel_text or "PERFORMANCE MAX" in channel_text:
        return "PMAX"
    if "SEARCH" in channel_text:
        return "SEARCH"
    return "OTHER"


def confirmed_primary_driver(signals: CampaignSignals) -> str:
    """Return a single confirmed driver among CPC/CVR/AOV."""

    def _score_issue() -> dict[str, float]:
        return {
            "CVR": abs(signals.dlog_cvr) if signals.dlog_cvr is not None and signals.dlog_cvr < 0 else 0.0,
            "AOV": abs(signals.dlog_aov) if signals.dlog_aov is not None and signals.dlog_aov < 0 else 0.0,
            "CPC": abs(signals.dlog_cpc) if signals.dlog_cpc is not None and signals.dlog_cpc > 0 else 0.0,
        }

    def _score_improve() -> dict[str, float]:
        return {
            "CVR": abs(signals.dlog_cvr) if signals.dlog_cvr is not None and signals.dlog_cvr > 0 else 0.0,
            "AOV": abs(signals.dlog_aov) if signals.dlog_aov is not None and signals.dlog_aov > 0 else 0.0,
            "CPC": abs(signals.dlog_cpc) if signals.dlog_cpc is not None and signals.dlog_cpc < 0 else 0.0,
        }

    scores = _score_issue() if signals.mode_tag == "ISSUE" else _score_improve()
    if max(scores.values()) > 0:
        return max(scores.items(), key=lambda item: (item[1], item[0]))[0]

    fallback_scores = {
        "CVR": abs(signals.dlog_cvr) if signals.dlog_cvr is not None else 0.0,
        "AOV": abs(signals.dlog_aov) if signals.dlog_aov is not None else 0.0,
        "CPC": abs(signals.dlog_cpc) if signals.dlog_cpc is not None else 0.0,
    }
    if max(fallback_scores.values()) > 0:
        return max(fallback_scores.items(), key=lambda item: (item[1], item[0]))[0]
    return "CPC"


def recommended_action(signals: CampaignSignals) -> str:
    mode_tag = signals.mode_tag
    driver = signals.primary_driver
    tag = signals.tag
    channel_type = channel_strategy_type(signals.channel)

    if tag == "LOW_VOL":
        return "광고 신호 부족: 판단 보류 후 High-intent 키워드/오디언스로 데이터 먼저 확보."
    if tag == "NEW":
        return "신규 캠페인: tROAS 가드레일 설정 후 High-intent 키워드/오디언스 중심으로 학습 안정화."
    if tag == "GONE":
        return "집행 중단 상태: 대체 캠페인 커버리지와 리마케팅 공백을 광고 운영 범위에서 점검."
    if tag == "UNDEFINED":
        return "지표 결측/0값: 태깅, 어트리뷰션, 집계 로직을 광고 데이터 관점에서 점검."

    if mode_tag == "ISSUE":
        if driver == "CPC":
            if channel_type == "SEARCH":
                return "Search CPC 이슈: 검색어 의도(브랜드/비브랜드·모델/일반) 재구성 후 매치타입·입찰을 차등 운영해 고CPC 비효율 유입을 축소."
            if channel_type == "PMAX":
                return "Pmax CPC 이슈: 저효율 Asset Group/상품군 예산을 축소하고 고효율군 중심으로 재배분해 유입단가를 안정화."
            return "CPC 이슈: High-intent 키워드/오디언스 비중 확대 + 비효율 검색어/지면 제외로 유입 효율 복원."
        if driver == "CVR":
            if channel_type == "SEARCH":
                return "Search CVR 이슈: 고의도 쿼리 중심으로 구조를 재편하고 광고 카피-랜딩 메시지 정합성을 강화해 전환율을 복원."
            if channel_type == "PMAX":
                return "Pmax CVR 이슈: 고CVR SKU 중심 Asset Group 재구성과 오디언스 시그널/신규고객 목표 정교화로 전환 질을 개선."
            return "CVR 이슈: 유입 의도(키워드/오디언스)와 랜딩 메시지 정합성을 맞춰 전환 효율 복원."
        if driver == "AOV":
            if channel_type == "SEARCH":
                return "Search AOV 이슈: 고가 SKU 쿼리 분리 운영과 tROAS 상향으로 고가 전환 비중을 확대하고 저객단가 유입은 축소."
            if channel_type == "PMAX":
                return "Pmax AOV 이슈: 고AOV SKU 전용 피드 라벨·Asset Group 분리와 value rule/tROAS 조정으로 고가 매출 중심 최적화."
            return "AOV 이슈: tROAS 상향/세분화 + 고가 SKU(예: S25 Ultra, 1TB) 유입 비중 확대."
        return "핵심 퍼널 지표 재점검 후 타게팅/입찰 규칙을 광고 운영 범위에서 재설정."

    if mode_tag in {"IMPROVE", "DEFENSE"}:
        if driver == "CPC":
            if channel_type == "SEARCH":
                return "Search CPC 개선 재현: 저CPC·고CTR 키워드군을 점진 증액하되 검색어 정제 룰을 유지해 효율을 확장."
            if channel_type == "PMAX":
                return "Pmax CPC 개선 재현: 저CPC Asset Group 중심 증액과 피드/검색테마 분리 유지로 효율 훼손 없이 스케일."
            return "CPC 개선 재현: 저CPC·고CTR 세그먼트 중심으로 점진 증액하며 효율 유지."
        if driver == "CVR":
            if channel_type == "SEARCH":
                return "Search CVR 개선 재현: 고CVR 쿼리 클러스터를 증액하고 승자 카피·랜딩 조합을 동일 의도군으로 복제 확장."
            if channel_type == "PMAX":
                return "Pmax CVR 개선 재현: 고CVR Asset Group 증액과 오디언스 시그널/신규고객 목표 유지로 전환 효율을 재현."
            return "CVR 개선 재현: 고CVR 키워드/오디언스와 랜딩 조합을 복제 확장."
        if driver == "AOV":
            if channel_type == "SEARCH":
                return "Search AOV 개선 재현: 고가 SKU 키워드/오디언스 비중을 확대하고 가치기반 입찰로 고단가 전환 점유율을 유지."
            if channel_type == "PMAX":
                return "Pmax AOV 개선 재현: 고AOV 피드 라벨 기반 증액과 value rule/tROAS 유지로 고가 매출 기여를 확장."
            return "AOV 개선 재현: tROAS 기반으로 고가 SKU 유입을 확대해 가치 높은 전환 비중 유지."
        if mode_tag == "DEFENSE":
            return "방어 운영: ROAS 유지 구간 중심으로 예산 재배분, 효율 악화 세그먼트는 축소."
        return "개선 패턴 재현을 위한 타게팅/입찰/소재 테스트 운영."

    return "No action rule matched."


def action_checklist(signals: CampaignSignals) -> str:
    mode_tag = signals.mode_tag
    driver = signals.primary_driver
    tag = signals.tag
    channel_type = channel_strategy_type(signals.channel)

    if tag == "LOW_VOL":
        return "1) 클릭/주문 최소 표본 확보; 2) High-intent 키워드/오디언스 우선 집행; 3) 성급한 입찰/구조 변경 제한"
    if tag == "NEW":
        return "1) tROAS 또는 가치기반 입찰 세팅; 2) 브랜드+모델+용량(예: S25 Ultra, 1TB) 키워드 우선; 3) 예산 증액 10-20% 단계 적용"
    if tag == "GONE":
        return "1) 중단 의도 확인; 2) 대체 캠페인 커버리지 확인; 3) 리마케팅 공백 점검"
    if tag == "UNDEFINED":
        return "1) 클릭/주문/매출 0값 점검; 2) 태깅/어트리뷰션 상태 확인; 3) 집계 기간 정합성 확인"

    if driver == "CPC":
        if mode_tag == "ISSUE":
            if channel_type == "SEARCH":
                return "1) 검색어 리포트 기반 저의도/저ROAS 쿼리 제외 및 네거티브 확장; 2) 브랜드/비브랜드·모델/일반 구조 분리 후 입찰 차등; 3) 고효율 키워드군으로 예산 재배분하며 CPC/CTR 가드레일 운영"
            if channel_type == "PMAX":
                return "1) Asset Group/상품군별 CPC·ROAS 분해; 2) 저효율 그룹 축소, 고효율 그룹 증액; 3) 검색테마·브랜드 제외·URL 확장 규칙 점검으로 무의도 유입 축소"
            return "1) Search term에서 저의도 쿼리 제외/네거티브 확장; 2) CTR(IMP/CLICK)·CPC 동시 점검해 고CPC·저CTR 구간 축소; 3) High-intent 키워드/오디언스 비중 확대"
        if channel_type == "SEARCH":
            return "1) 저CPC·고CTR 키워드군 10-20% 단계 증액; 2) CPC 급등/품질점수 하락 알림 운영; 3) 검색어 정제 룰 유지하며 확장"
        if channel_type == "PMAX":
            return "1) 저CPC Asset Group 단계 증액; 2) 크리에이티브 피로도 점검 및 승자 에셋 재배치; 3) 피드/검색테마 분리 규칙 유지"
        return "1) 저CPC·고CTR 세그먼트 예산 확대; 2) 상위 효율 쿼리 점유율 손실 여부 점검; 3) CPC 급등 알림/입찰 상한 가드레일 유지"
    if driver == "CVR":
        if mode_tag == "ISSUE":
            if channel_type == "SEARCH":
                return "1) 브랜드+모델+용량 등 고의도 쿼리 비중 확대, 저의도 쿼리 제외; 2) 쿼리 의도별 RSA 메시지와 랜딩 매핑 정합성 강화; 3) tCPA/tROAS 목표를 단계 조정해 학습 안정화 후 재상향"
            if channel_type == "PMAX":
                return "1) 고CVR SKU 기준 Asset Group 재구성; 2) 리마케팅/고LTV 중심 오디언스 시그널 및 신규고객 목표 재정의; 3) 전환가치·전환이벤트 집계 우선순위 점검"
            return "1) 키워드/오디언스 의도와 랜딩 메시지 일치 점검; 2) CVR 낮은 검색어·오디언스는 제외/입찰 하향하고 고CVR 세그먼트로 예산 재배분; 3) 전환 가능성 낮은 유입 지면·매체 비중 축소"
        if channel_type == "SEARCH":
            return "1) 고CVR 쿼리/오디언스 10-20% 증액; 2) 승자 카피·랜딩 조합 복제 및 테스트 슬롯 분리; 3) 스케일 구간 CVR 하락 가드레일 운영"
        if channel_type == "PMAX":
            return "1) 고CVR Asset Group 우선 증액; 2) 신규고객 가치 규칙/오디언스 시그널 유지; 3) 소재 교체는 단계 적용해 학습 리셋 최소화"
        return "1) 고CVR 키워드/오디언스 확장; 2) 승자 랜딩/소재 조합 복제; 3) 스케일 시 CVR 하락 가드레일 설정"
    if driver == "AOV":
        if mode_tag == "ISSUE":
            if channel_type == "SEARCH":
                return "1) 고가 SKU 키워드군 분리 및 tROAS 목표 상향; 2) 저객단가 쿼리 매치타입 제한/입찰 하향; 3) 고가 모델 중심 카피/오디언스로 예산 재배분"
            if channel_type == "PMAX":
                return "1) 고AOV SKU 피드 라벨/Asset Group 분리; 2) value rule로 고가 전환 가치 가중; 3) 저AOV SKU 비중 축소 및 예산 재배분"
            return "1) tROAS 목표 상향/세분화; 2) 고가 SKU 키워드/오디언스(예: S25 Ultra, 1TB) 비중 확대; 3) 저객단가 쿼리/지면 제외 및 입찰 하향"
        if channel_type == "SEARCH":
            return "1) 고AOV 키워드/오디언스 단계 증액; 2) 고가 SKU 애드그룹 점유율 유지; 3) AOV/ROAS 하락 가드레일로 스케일 관리"
        if channel_type == "PMAX":
            return "1) 고AOV 피드 라벨 Asset Group 증액; 2) value rule/tROAS 유지로 고가 전환 우선 탐색; 3) 할인 중심 저가 유입 비중 과다 여부 점검"
        return "1) tROAS 유지로 고가 구매 유저 탐색 확대; 2) 고AOV SKU/번들 키워드 점유율 유지; 3) 할인 중심 트래픽 유입 과다 여부 점검"

    return "1) 타게팅/입찰/크리에이티브 기본 세팅 점검; 2) 트래픽 품질 지표 점검; 3) 예산 배분 재검토"
