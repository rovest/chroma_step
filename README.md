# ChromaStep 3D

ChromaStep 3D는 2D 이미지에서 특정 색상 영역을 분리하고, 각 영역을 서로 다른 높이로 압출해 2.5D 형태의 3D 모델(STEP/STL)로 내보내는 데스크톱 앱입니다.

- GUI: PyQt5
- 이미지 처리: OpenCV + NumPy
- CAD/Export: FreeCAD Python API (headless)

---

## 1. 주요 기능

- 원본 이미지 표시
- 레이어(Gray / Light Blue / Dark Navy)별 색상 허용 오차(Tolerance) 조절
- 실시간 마스크 프리뷰
- 레이어별 목표 높이(mm) 설정
- 백그라운드 스레드(QThread) 기반 3D 생성(진행률/상태 표시)
- STEP / STL 파일 저장

---

## 2. 프로젝트 구조

```text
chroma_step/
├── main.py
├── core/
│   ├── __init__.py
│   ├── color_processor.py
│   └── cad_builder.py
└── ui/
    ├── __init__.py
    ├── main_window.py
    └── worker.py
```

- `core/color_processor.py`: HSV 마스킹, 컨투어 추출, 레이어 DTO 생성
- `ui/main_window.py`: 메인 UI, 이미지 로딩, 프리뷰, 워커 실행/종료 제어
- `ui/worker.py`: FreeCAD 기반 실제 3D 솔리드 생성 및 STEP/STL 내보내기

---

## 3. 요구 환경

- Python 3.10+
- PyQt5 (5.15+)
- OpenCV-Python
- NumPy
- FreeCAD Python 모듈 (`FreeCAD`, `Part`, `MeshPart`)

Linux(예: Ubuntu)에서는 일반적으로 `freecad-python3` 패키지가 필요합니다.

---

## 4. 설치

### 4.1 Python 패키지 설치

```bash
pip install PyQt5 opencv-python numpy
```

### 4.2 FreeCAD 설치

#### Ubuntu 계열

```bash
sudo apt-get update
sudo apt-get install -y freecad-python3
```

#### Windows

1. FreeCAD 설치
2. FreeCAD `bin` 경로 확인 (예: `C:\Program Files\FreeCAD 0.21\bin`)

---

## 5. FreeCAD 경로 설정 (중요)

`ui/worker.py`는 아래 변수로 FreeCAD 모듈 경로를 찾습니다.

```python
FREECAD_BIN_PATH = os.environ.get("FREECAD_BIN_PATH", "/usr/lib/freecad-python3/lib")
```

환경에 맞게 아래 중 하나를 사용하세요.

### 방법 A) 환경변수 사용 (권장)

Linux:

```bash
export FREECAD_BIN_PATH=/usr/lib/freecad-python3/lib
```

Windows PowerShell:

```powershell
$env:FREECAD_BIN_PATH = "C:\Program Files\FreeCAD 0.21\bin"
```

### 방법 B) 코드 상수 직접 수정

`ui/worker.py`의 `FREECAD_BIN_PATH` 기본값을 직접 수정.

---

## 6. 실행

프로젝트 루트에서:

```bash
python3 main.py
```

---

## 7. 사용 방법 (처음 사용자 기준)

1. `Load Image...` 클릭 후 이미지 선택
2. 좌측 상단에 원본 이미지, 좌측 하단에 마스크 프리뷰 확인
3. 각 레이어에서:
   - `Height(mm)` 조절
   - `Color Tolerance` 슬라이더 조절
4. 마스크 분리가 원하는 형태인지 확인
5. `Generate 3D Solid Model` 클릭
6. 하단 진행률/상태 메시지 확인
7. 완료 팝업에서 생성된 STEP/STL 경로 확인

### 레이어 의미

- Layer 1: Base (Gray)
- Layer 2: Middle (Light Blue)
- Layer 3: Top (Dark Navy)

현재 로직은 **절대 높이 압출(Absolute Extrusion)** 방식입니다.
- 각 레이어는 `Z=0`에서 시작하여 설정 높이(`z_height`)까지 압출됩니다.
- 이후 Boolean Union으로 하나의 솔리드로 병합됩니다.

---

## 8. 출력 파일

기본 출력 경로: 앱 실행 당시 작업 디렉터리

파일명 형식:

- `파일명_YYYYMMDD_HHMMSS.step`
- `파일명_YYYYMMDD_HHMMSS.stl`

---

## 9. 스레드/중단 동작

- 3D 생성은 `QThread`(`CADWorker`)에서 실행되어 UI 멈춤을 최소화합니다.
- 앱 종료 시 생성 중이라면 `requestInterruption()` 후 최대 2초 대기 후 종료합니다.

---

## 10. 자주 발생하는 문제

### 10.1 `No module named FreeCAD` / `Part` / `MeshPart`

- FreeCAD 설치 확인
- `FREECAD_BIN_PATH` 설정 확인
- 현재 셸에서 아래 확인:

```bash
python3 -c "import sys, os; print(os.environ.get('FREECAD_BIN_PATH')); print(sys.path)"
```

### 10.2 `No contours provided for CAD generation`

- 이미지가 로드되었는지 확인
- Tolerance가 너무 좁아 해당 레이어 색상이 검출되지 않을 수 있음
- 레이어별 슬라이더를 조절해 마스크 프리뷰에서 흰색 영역이 보이는지 먼저 확인

### 10.3 생성이 중단되거나 실패

- 매우 복잡한 컨투어는 Boolean 연산이 불안정할 수 있음
- 더 단순한 입력 이미지로 먼저 테스트
- Tolerance를 조정해 노이즈를 줄인 뒤 재시도

---

## 11. 개발 메모

- 워커 입력 스키마(통합 DTO):

```python
[
  {
    "layer_name": str,
    "z_height": float,
    "contours": list,
  }
]
```

- 색상 분리/컨투어 추출은 `core/color_processor.py`
- CAD 생성/Boolean/Export는 `ui/worker.py`

---

## 12. 향후 개선 아이디어

- 레이어 추가/삭제 동적 지원
- 레이어별 색상 선택기 및 자동 색상 샘플링
- 출력 경로 선택 UI
- 모델 미리보기(3D viewport)
- 테스트 코드(마스킹/DTO/워커 단위 테스트)
