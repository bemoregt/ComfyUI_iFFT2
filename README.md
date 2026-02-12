# ComfyUI Custom Node — Spectrum to Image (iFFT2D)

진폭 스펙트럼 이미지와 위상 스펙트럼 이미지를 입력받아 2D 역 FFT(iFFT)로 원본 이미지를 복원하는 ComfyUI 커스텀 노드입니다.

[bemoregt/ComfyUI_CustomNode_Image2Spectrum](https://github.com/bemoregt/ComfyUI_CustomNode_Image2Spectrum) 의 **역연산 노드**입니다.

---

## 노드 정보

| 항목 | 값 |
|------|----|
| 노드 이름 | Spectrum to Image (iFFT2D) |
| 카테고리 | image/transform |
| 클래스명 | Spectrum2Image |

---

## 입력 / 출력

### 입력

| 이름 | 타입 | 설명 |
|------|------|------|
| `amplitude_spectrum` | IMAGE | 진폭 스펙트럼 이미지 (Image_Spectrum 노드 출력) |
| `phase_spectrum` | IMAGE | 위상 스펙트럼 이미지 (Image_Spectrum 노드 출력) |
| `amplitude` | INT (1–50) | Image_Spectrum 노드에서 사용한 amplitude 값과 동일하게 설정 (기본값: 20) |

### 출력

| 이름 | 타입 | 설명 |
|------|------|------|
| `reconstructed_image` | IMAGE | iFFT로 복원된 이미지 (RGB, float32) |

---

## 동작 원리

### 정방향 (Image_Spectrum)

```
grayscale → fft2 → fftshift → 진폭: A·log1p(|F|)/255  /  위상: (φ+π)/(2π)
```

### 역방향 (Spectrum2Image, 본 노드)

```
진폭 역정규화: |F| = expm1(amp × 255 / A)
위상 역정규화: φ  = phase_img × 2π − π
복소수 재구성: F  = |F| · e^(jφ)
역 FFT 수행:   image = real( ifft2( ifftshift(F) ) )
```

---

## 설치

### 방법 1 — 직접 복사

```bash
cp -r ComfyUI_CustomNode_FFT2_iFFT2 <ComfyUI 경로>/custom_nodes/
```

### 방법 2 — git clone

```bash
cd <ComfyUI 경로>/custom_nodes
git clone https://github.com/bemoregt/ComfyUI_CustomNode_FFT2_iFFT2.git
```

의존 패키지 설치:

```bash
pip install -r custom_nodes/ComfyUI_CustomNode_FFT2_iFFT2/requirements.txt
```

ComfyUI를 재시작하면 노드 목록에 **"Spectrum to Image (iFFT2D)"** 가 표시됩니다.

---

## 사용 예시

```
[Load Image]
     │
[Image_Spectrum]  (amplitude=20)
  │          │
[amp_img]  [phase_img]
  │          │
[Spectrum2Image]  (amplitude=20)
     │
[복원된 이미지]
```

> `amplitude` 값은 두 노드에서 **반드시 동일**해야 올바르게 복원됩니다.

---

## 주의사항

- 진폭/위상 이미지가 `float32`로 저장되는 과정에서 미세한 양자화 오차가 발생할 수 있어 완벽한 픽셀 단위 복원은 아닐 수 있습니다.
- 입력 이미지가 RGB인 경우 첫 번째 채널(R)만 사용합니다 (정방향과 동일하게 그레이스케일 처리).
- 출력 이미지는 그레이스케일을 RGB 3채널로 확장한 형태입니다.

---

## 의존성

- Python 3.8+
- torch
- numpy
- scipy
- Pillow

---

## 라이선스

MIT License
