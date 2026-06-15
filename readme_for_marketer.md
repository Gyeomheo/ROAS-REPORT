# ROAS 리포트 자동화 초간단 가이드

## 1) 이게 뭐예요?
작년 같은 기간과 비교해서 **ROAS가 왜 변했는지** 자동으로 요약해주는 주간 리포트 도구입니다.

## 2) 이 도구가 해주는 일
- 엑셀 성과 데이터를 자동 정리
- ROAS 하락 이슈 Top 항목 추출
- ROAS 개선 포인트 Top 항목 추출
- 원인 드라이버를 `CPC / CVR / AOV`로 표시
- 바로 실행할 액션 문구와 체크리스트 제공
- 결과물을 JSON / HTML / Excel로 생성

## 3) 실행 전에 준비할 것
- 원본 Excel 파일 위치 확인 (실행 시 파일 선택창에서 고름)
- Python 3.10+
- 권장 패키지: `polars`, `openpyxl`

## 4) 실행 방법

### 클렌징만 먼저 실행

VS Code 터미널에서:

```bash
cd marketing_orchestrator
python cleansing.py
```

파일 선택창에서 원본 Excel을 고르면 `output/GMPD RAW_Cleaned_YYYYMMDD_HHMMSS.xlsx`만 생성됩니다.

### 이슈분석만 따로 실행

클렌징 결과 파일을 선택해서 이슈분석/리포트만 실행:

```bash
python main.py --mode analyze
```

파일 선택창에서 `output/GMPD RAW_Cleaned_...xlsx`를 선택합니다.

### 전체 한 번에 실행

```bash
cd marketing_orchestrator
python main.py
```

연도 지정 실행:
```bash
python main.py --curr-year 2026 --prev-year 2025
```

## 5) 결과 파일 위치
- `marketing_orchestrator/output/summary.html` (공유용 화면)
- `marketing_orchestrator/output/summary.xlsx` (실무 작업용)
- `marketing_orchestrator/output/summary.json` (시스템/추적용)

## 6) 결과를 30초 만에 읽는 법
- `Impact_adj < 0`: ROAS 하락 기여 (우선 대응)
- `Impact_adj > 0`: ROAS 상승 기여 (재현/확대)
- `primary_driver = CPC`: 유입 단가 이슈/개선
- `primary_driver = CVR`: 전환율 이슈/개선
- `primary_driver = AOV`: 객단가 이슈/개선
- `tag = LOW_VOL`: 표본이 작아 해석 주의

## 7) 참고(자동 적용 규칙)
- OBJECTIVE는 `CONVERSION`만 사용
- DIVISION은 `MX`, `VD`, `DA`만 사용
- MTD 기준으로 올해/전년을 같은 월/일 범위로 맞춰 비교

## 8) 한 줄 운영 팁
매주 `summary.html`로 전체 공유하고, 실행 담당자는 `summary.xlsx`의 `issues`/`improvements` 시트를 바로 액션 관리용으로 쓰면 됩니다.
