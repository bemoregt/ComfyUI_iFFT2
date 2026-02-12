import torch
import numpy as np
from scipy.fft import ifft2, ifftshift


class Spectrum2Image:
    """
    ComfyUI Custom Node: Spectrum2Image

    진폭 스펙트럼 이미지와 위상 스펙트럼 이미지를 입력받아
    2D 역 FFT(iFFT)를 수행하여 원본 이미지를 복원합니다.

    ※ amplitude 값은 Image_Spectrum 노드의 amplitude 값과 반드시 동일해야 합니다.

    Image_Spectrum 노드 (bemoregt/ComfyUI_CustomNode_Image2Spectrum) 의
    역연산 노드입니다.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "amplitude_spectrum": ("IMAGE",),
                "phase_spectrum": ("IMAGE",),
                "amplitude": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
                "normalize": ("BOOLEAN", {
                    "default": False,
                    "label_on": "ON (출력 범위 자동 정규화)",
                    "label_off": "OFF (원본 밝기 그대로)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reconstructed_image",)
    FUNCTION = "spectrum_to_image"
    CATEGORY = "image/transform"

    def spectrum_to_image(self, amplitude_spectrum, phase_spectrum, amplitude, normalize):
        # 텐서에서 numpy 배열 추출 (배치 첫 번째 요소)
        amp_np = amplitude_spectrum[0].cpu().numpy()  # [H, W] 또는 [H, W, C]
        phase_np = phase_spectrum[0].cpu().numpy()    # [H, W] 또는 [H, W, C]

        # 다채널(RGB 등)인 경우 첫 번째 채널(그레이스케일)만 사용
        if amp_np.ndim == 3:
            amp_np = amp_np[:, :, 0]
        if phase_np.ndim == 3:
            phase_np = phase_np[:, :, 0]

        # ── 진폭 스펙트럼 역정규화 ──────────────────────────────────────────
        # 정방향: amp_img = amplitude * log1p(|F_shift|) / 255.0
        # 역방향: |F_shift| = expm1(amp_np * 255.0 / amplitude)
        # ※ amplitude 불일치 시 magnitude가 수천 배 작아져 검정 이미지가 됩니다.
        exponent = amp_np.astype(np.float64) * 255.0 / amplitude
        # 오버플로우 방지 (exp(709) ≈ float64 최대값)
        exponent = np.clip(exponent, 0, 700)
        magnitude = np.expm1(exponent)
        # NaN / Inf 방어
        magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

        # ── 위상 스펙트럼 역정규화 ──────────────────────────────────────────
        # 정방향: phase_img = (phase + π) / (2π)   →  [0, 1]
        # 역방향: phase = phase_img * 2π - π        →  [-π, π]
        phase_rad = phase_np.astype(np.float64) * 2.0 * np.pi - np.pi

        # ── 복소 주파수 도메인 재구성 ─────────────────────────────────────
        f_shift = magnitude * np.exp(1j * phase_rad)

        # ── 역 FFT ────────────────────────────────────────────────────────
        f_transform = ifftshift(f_shift)
        reconstructed = np.real(ifft2(f_transform))

        # ── 이미지 후처리 ─────────────────────────────────────────────────
        if normalize:
            # 출력 범위 자동 정규화: amplitude 불일치가 있어도 이미지가 보임
            r_min, r_max = reconstructed.min(), reconstructed.max()
            if r_max > r_min:
                reconstructed = (reconstructed - r_min) / (r_max - r_min)
            else:
                reconstructed = np.zeros_like(reconstructed)
        else:
            # 원본 밝기 기준: [0, 255] → [0, 1]
            reconstructed = np.clip(reconstructed, 0, 255) / 255.0

        result = reconstructed.astype(np.float32)

        # ComfyUI IMAGE 포맷: [B, H, W, C] (RGB 3채널로 확장)
        result_rgb = np.stack([result] * 3, axis=-1)          # [H, W, 3]
        result_tensor = torch.from_numpy(result_rgb)[None,]   # [1, H, W, 3]

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "Spectrum2Image": Spectrum2Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Spectrum2Image": "Spectrum to Image (iFFT2D)",
}
