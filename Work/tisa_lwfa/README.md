# Ti:Sapphire Laser Wakefield Acceleration (LWFA)

## Physics

800 nm Ti:Sa レーザーパルスが underdense プラズマ（n << nc）中を伝播し、プラズマ波（wake）を励起する。電子ビームがこの wake に捕捉され、高エネルギーに加速される。

| パラメータ | 値 |
|---|---|
| レーザー波長 | 800 nm (Ti:Sapphire) |
| 規格化振幅 a0 | ~4 (e_max = 16 TV/m) |
| パルス幅 | 15 fs |
| プラズマ密度 | 2×10²³ m⁻³ (= 2×10¹⁷ cm⁻³) |
| n/nc | 0.00115 |
| プラズマ波長 λp | ~23.6 μm |
| グリッド | 64×512 (AMR level=1) |

## 観測できる現象

- `Ez` フィールド：移動窓と一緒に進むプラズマ wake 構造
- 電子ビーム（beam species）の加速と phase space の変化
- a0 > 1 では非線形（bubble/blowout）regime に移行

## 実行方法

```bash
cd Work/tisa_lwfa
mpirun -n 1 ../../build_cuda/bin/warpx.2d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_2d_tisa_lwfa
```

CPU版:
```bash
mpirun -n 4 ../../build/bin/warpx.2d.MPI.OMP.DP.PDP.OPMD.EB.QED inputs_2d_tisa_lwfa
```

## 可視化 (openPMD-viewer)

```bash
pip install openpmd-viewer matplotlib
```

```python
from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt

ts = OpenPMDTimeSeries('./diag1/')
Ez, info = ts.get_field('Ez', iteration=ts.iterations[-1])
plt.imshow(Ez.T, aspect='auto', origin='lower',
           extent=[info.x[0]*1e6, info.x[-1]*1e6,
                   info.z[0]*1e6, info.z[-1]*1e6])
plt.xlabel('x (μm)'); plt.ylabel('z (μm)')
plt.title('Ez - Wakefield'); plt.colorbar(); plt.show()
```

## 参考

- Tajima & Dawson (1979), PRL 43, 267
- Yano et al. (2018), Phys. Plasmas 25, 103104 — LWFA + Hawking-like effects
