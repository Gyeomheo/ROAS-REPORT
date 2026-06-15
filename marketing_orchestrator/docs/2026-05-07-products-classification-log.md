# 2026-05-07 Products 분류 로직 정밀 개선

- 원본: `C:\Users\Heo Gyeom\OneDrive - Artience Inc\바탕 화면\sec campaign name.xlsx`
- 분석: `캠페인`의 `CN~([a-z0-9]+)` 코드를 추출하고 `키워드` 분포로 제품군 역추적.
- 커버리지: 파일 내 `CN~` 코드 81개 전체 매핑 (`missing []`).
- taxonomy: 파일상 제품군 우선 + 기존 `PRODUCTS` canonical label 재사용.

## 주요 매핑
- `arc`, `ekarc`, `ebarcm`, `ebarcp` -> `AIR CONDITIONER`
- `acl`, `ekacl`, `ebaclm`, `ebaclp` -> `AIR PURIFIER`
- `ard`, `ekard`, `ebardm`, `ebardp` -> `AIR DRESSER/SHOE DRESSER`
- `jbl`, `ekjbl`, `ebhmkm`, `ebhmkp` -> `HARMAN`
- `qoop`, `ebqoom`, `ebqoop` -> `MICROWAVE/OTR/QOOKER`
- `emvip`, `ebmvim`, `ebmvip`, `ekmni` -> `LIFESTYLE TV`
- `ekkch`, `ebkchm`, `ebkchp` -> `REFRIGERATOR`
- `ekpri`, `ebprim`, `ebprip` -> `PRINTER`

## 검증
- `py_compile src\ingestion.py tests\test_regressions.py` 통과
- `workbook_codes 81`, `mapping_entries 90`, `missing []`
- 샘플: `arc AIR CONDITIONER`, `ekjbl HARMAN`, `ebmvip LIFESTYLE TV`, `ekkch REFRIGERATOR`
- 전체 unittest는 현재 런타임에 `polars`가 없어 미실행.
