# Ti:Sapphire Laser Hole Boring & Ion Acceleration

## Physics

高強度 Ti:Sa レーザー（a0 = 16）が overdense 水素フォイル（30 nc）に当たり、放射圧でプラズマを押し込む（hole boring）。同時にイオンが前方に加速される（TNSA/RPA）。

| パラメータ | 値 |
|---|---|
| レーザー波長 | 800 nm (Ti:Sapphire) |
| 規格化振幅 a0 | 16 |
| パルス幅 | 30 fs |
| ターゲット密度 | 30 nc = 5.22×10²⁸ m⁻³ |
| ターゲット厚さ | 5 μm（フラットフォイル + pre-plasma） |
| グリッド | 384×512 |

**Hole boring 速度の見積もり:**

```
v_HB/c = sqrt(I / (rho * c^3)) = sqrt(a0^2 * nc / n0) ~ 0.1c
```

## 観測できる現象

- イオン加速スペクトル（`histuH` reduced diagnostic）
- 前方・後方の粒子分離（filter by uz > 0 / uz < 0）
- ターゲット面の変形・加速（rho フィールド）
- pre-plasma での吸収 vs 本体への浸透

## Input files

| ファイル | ジオメトリ | 備考 |
|---|---|---|
| `inputs_2d_tisa_hole_boring` | 2D slab | 元版 |
| `inputs_rz` | RZ quasi-3D | 推奨。円筒対称ターゲットに正しい |

RZ版は dt=0.045fs と非常に細かいため、1000steps ≈ 25分で 45fs しかカバーしない。
v_HB ≈ 0.048c なので 45fs で ~0.65µm の穿孔。意味ある可視化には 5000 steps (125分) 推奨。

## 実行方法

```bash
cd Work/tisa_hole_boring

# 2D slab
../../build_cuda/bin/warpx.2d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_2d_tisa_hole_boring

# RZ quasi-3D
../../build_cuda/bin/warpx.rz.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_rz
```

## 可視化

```python
from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt

ts = OpenPMDTimeSeries('./diagInst/')
rho, info = ts.get_field('rho_hydrogen', iteration=ts.iterations[-1])
plt.imshow(rho.T, aspect='auto', origin='lower')
plt.title('Hydrogen density'); plt.colorbar(); plt.show()
```

イオンエネルギースペクトル（CSV）:
```python
import numpy as np
data = np.loadtxt('diags/reducedfiles/histuH.txt', skiprows=1)
# columns: step, time, bin_centers..., counts...
```

## 参考

- Yano et al. (2019), arXiv:1904.09057 — Hole boring & relativistic transparency
- Robinson et al. (2009), New J. Phys. 11, 083018
