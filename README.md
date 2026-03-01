# ChromaStep 3D

ChromaStep 3D는 2D 이미지를 색상별로 분리해 각 레이어를 서로 다른 높이로 압출하고, 결과를 STEP/STL로 저장하는 PyQt5 기반 데스크톱 앱입니다.

현재 버전은 다음에 초점을 둡니다.
- 자동 색상 클러스터링 + 동적 레이어 UI
- 홀(hole) 보존 컨투어 추출 (`RETR_CCOMP`)
- FreeCAD 기반 2.5D 압출
- 개별 솔리드 메싱(거대 Boolean 회피)
- STL 저장 후 PyVista/VTK 후처리 힐링
- GUI 내 2D 마스크 / 3D 결과 뷰 전환

---

## 1. 주요 기능

- 원본 이미지 표시
- 로드 시 이미지 해상도 검증 후, 초과 시 자동 다운스케일 여부 확인
- 색상 자동 분석(k-means) 후 레이어 동적 생성
- 레이어별 높이(mm), tolerance 조정
- 2D 마스크 프리뷰 + 3D 뷰어(PyvistaQt)
- 백그라운드 스레드(QThread) 기반 CAD 생성
- STEP/STL/metadata 파일 저장

---

## 2. 프로젝트 구조

```text
chroma_step/
├── main.py
├── README.md
├── core/
│   ├── __init__.py
│   ├── cad_builder.py            # 레거시/참고용
│   └── color_processor.py        # 현재 색상 분석/컨투어 추출 핵심
└── ui/
    ├── __init__.py
    ├── main_window.py            # 메인 UI, 2D/3D 뷰 전환
    └── worker.py                 # FreeCAD CAD 생성/저장 스레드
```

실제 생성 파이프라인은 `ui/worker.py` + `core/color_processor.py`를 사용합니다.

---

## 3. 요구 환경

- Python 3.10+
- PyQt5 (5.15+)
- NumPy
- OpenCV (`opencv-python-headless` 권장)
- PyVista, pyvistaqt
- FreeCAD Python 모듈 (`FreeCAD`, `Part`, `MeshPart`, `Mesh`)

Linux(Ubuntu) 기준 FreeCAD:
```bash
sudo apt-get update
sudo apt-get install -y freecad-python3
```

---

## 4. 권장 설치 절차

프로젝트 루트에서:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyvista pyvistaqt "opencv-python-headless>=4.8,<4.9" "numpy>=1.26,<2" PyQt5
```

`--system-site-packages`를 쓰는 이유:
- 시스템 FreeCAD Python 모듈과의 ABI/Qt 충돌을 줄이기 위해

---

## 5. FreeCAD 경로 설정

`ui/worker.py`는 `FREECAD_BIN_PATH`와 `PYTHONPATH`를 순서대로 확인해
FreeCAD 라이브러리 경로를 찾습니다.

```bash
export FREECAD_BIN_PATH=/usr/lib/freecad-python3/lib
```

또는:
```bash
export PYTHONPATH=/usr/lib/freecad-python3/lib:$PYTHONPATH
```

Windows 예시:
```powershell
$env:FREECAD_BIN_PATH = "C:\Program Files\FreeCAD 0.21\bin"
```

---

## 6. 실행

```bash
cd /home/paddlesdesktop/Workspaces/chroma_step
source .venv/bin/activate
export FREECAD_BIN_PATH=/usr/lib/freecad-python3/lib
python main.py
```

---

## 7. 사용자 워크플로우

1. `Load Image...` 클릭
2. 해상도 검사
   - `width > 640` 또는 `height > 640`이면 자동 다운스케일 여부 확인
3. 자동 색상 분석 수행
4. 우측 레이어 UI가 클러스터 수에 맞게 동적 생성
5. 레이어별 Height/Tolerance 조정
   - 높이 스핀의 버튼 step은 `1.2mm`
   - 수동 타이핑 값은 그대로 허용
6. `Generate 3D Solid Model` 실행
7. 완료 후:
   - 좌측 3D 뷰로 자동 전환
   - 결과 폴더 열기 가능
   - Recent Results 목록에 최근 작업(최대 5개) 기록

---

## 8. 현재 데이터 스키마

### 8.1 컨투어 DTO (`color_processor -> worker`)

```python
[
  {
    "layer_name": str,
    "z_height": float,
    "shapes": [
      {
        "outer": [(x, y), ...],
        "holes": [[(x, y), ...], ...],
      },
      ...
    ],
  },
  ...
]
```

### 8.2 색상 분석 반환 (`analyze_image_colors`)

```python
[
  {
    "color_bgr": (b, g, r),
    "color_hsv": (h, s, v),
    "tolerance": int,
  },
  ...
]
```

---

## 9. CAD/메싱 파이프라인 (현재 동작)

1. 각 레이어 shape를 Face 생성
   - `Face([outer_wire] + hole_wires)` 시도
   - 실패 시 `Face(outer_wire)` fallback
2. `Z=0`에서 `z_height`까지 개별 압출
3. 유효성 필터
   - `solid.isValid()`
   - `solid.Volume > 1.0`
4. STEP
   - 솔리드들을 `Part::Feature`로 문서에 올려 export
5. STL
   - 개별 솔리드별 `MeshPart.meshFromShape(..., LinearDeflection, AngularDeflection)`
   - `final_mesh.addMesh(...)`로 누적
6. FreeCAD 메쉬 힐링 체인
   - duplicated points/facets, degenerated facets, non-manifolds, holes
7. Post-Export VTK 힐링
   - `pv.read -> clean -> fill_holes -> clean().triangulate() -> overwrite`

---

## 10. 출력 파일 구조

기본 루트:
```text
~/ChromaStep/exports/
```

작업별 폴더:
```text
YYYY-MM-DD_HH-MM-SS_<image_stem>/
```

예시:
```text
~/ChromaStep/exports/2026-02-27_11-59-10_logo/
├── final_relief.step
├── final_relief.stl
├── metadata.json
└── error.log            # 실패 시 생성
```

---

## 11. 문제 해결

### 11.1 `No module named FreeCAD`
- FreeCAD 시스템 패키지 설치 확인
- `FREECAD_BIN_PATH` 확인

### 11.2 Qt/xcb plugin 충돌
- `main.py`는 OpenCV Qt plugin 경로를 정리하도록 구성됨
- 가능하면 `opencv-python-headless` 사용

### 11.3 STL이 비정상(구멍/깨짐)
- 레이어 tolerance를 과도하게 넓히지 않기
- 이미지 노이즈/아주 작은 텍스트를 줄인 입력 사용
- metadata의 `valid_solid_count`, `meshed_solid_count` 확인

### 11.4 PyVista 렌더 실패
- STL 로드 후 `clean().triangulate()` 수행
- `n_points == 0`이면 표시 중단

---

## 12. 개발 참고

- 엔트리포인트: `main.py`
- UI 흐름: `ui/main_window.py`
- 백그라운드 CAD 처리: `ui/worker.py`
- CV/컨투어 추출: `core/color_processor.py`

