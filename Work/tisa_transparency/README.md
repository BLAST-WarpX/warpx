# Ti:Sapphire Relativistic Induced Transparency

## Physics

高強度レーザーが overdense プラズマに当たるとき、電子が相対論的速度になると有効質量が増加し、実効的な臨界密度が上昇する：

```
nc_eff = gamma * nc ≈ (a0 / sqrt(2)) * nc  [線形偏光の場合]
```

透過条件：`a0 > sqrt(2) * n/nc`

このファイルの設定（2×nc のプラズマ）では：`a0 > 2*sqrt(2) ≈ 2.83` で透過が起きる。

| パラメータ | 値 |
|---|---|
| レーザー波長 | 800 nm (Ti:Sapphire) |
| 初期 a0 | ~1 (反射側) |
| プラズマ密度 | 最大 2×nc = 3.48×10²⁷ m⁻³ |
| 密度プロファイル | 指数関数ランプ → 2nc 一様層 |
| グリッド | 256×128 |

## 実験方法

1. **まず a0~1 で実行** → レーザーが反射されるのを確認
2. **`laser1.e_max` を 12.e12 V/m（a0~3）に変更** → 透過が始まる
3. **`laser1.e_max` を 20.e12 V/m（a0~5）に変更** → 完全透過

```
# inputs ファイルの変更箇所:
laser1.e_max = 4.e12    # a0~1: 反射
laser1.e_max = 12.e12   # a0~3: 透過の閾値付近
laser1.e_max = 20.e12   # a0~5: 透過
```

## 実行方法

```bash
cd Work/tisa_transparency
mpirun -n 1 ../../build_cuda/bin/warpx.2d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_2d_tisa_transparency
```

## 可視化

```python
from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt

ts = OpenPMDTimeSeries('./diag1/')
for it in ts.iterations[::2]:
    Ey, info = ts.get_field('Ey', iteration=it)
    plt.plot(info.z*1e6, Ey[len(info.x)//2, :], label=f't={ts.t[ts.iterations==it][0]*1e15:.0f} fs')
plt.xlabel('z (μm)'); plt.ylabel('Ey (V/m)')
plt.legend(); plt.title('Laser field penetration'); plt.show()
```

## 参考

- Yano et al. (2019), arXiv:1904.09057 — Hole boring & relativistic transparency
- Palaniyappan et al. (2012), Nature Phys.
