"""
Animation generator for WarpX PhD simulations.
Each frame: top row = 2D colormap, bottom row = 1D line profile at x=0.
Laser propagates in +z direction.
"""
import os, json, glob, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = '/mnt/d/Dev/warpx/Work'
OUT  = '/mnt/d/Dev/warpx/Work/animations'
os.makedirs(OUT, exist_ok=True)


def read_field(json_path, field, component=None):
    """Return (arr[nz,nx], grid_dict). axisLabels=['z','x']."""
    with open(json_path) as f:
        d = json.load(f)
    it   = list(d['data'].values())[0]
    time = it['attributes']['time']['value']
    step = int(list(d['data'].keys())[0])
    fd   = it['fields'][field]
    src  = fd[component] if component else fd
    arr  = np.array(src['data'], dtype=np.float64)
    att  = fd['attributes']
    z0, x0 = att['gridGlobalOffset']['value']
    dz, dx = att['gridSpacing']['value']
    nz, nx = arr.shape
    return arr, dict(time=time, step=step,
                     z0=z0, x0=x0, dz=dz, dx=dx, nz=nz, nx=nx,
                     zmax=z0+nz*dz, xmax=x0+nx*dx)


def z_axis(g):
    return np.linspace(g['z0'], g['zmax'], g['nz']) * 1e6   # µm

def x_axis(g):
    return np.linspace(g['x0'], g['xmax'], g['nx']) * 1e6

def extent_um(g):
    return [g['z0']*1e6, g['zmax']*1e6, g['x0']*1e6, g['xmax']*1e6]

def ix0(g):
    """Index of x=0 (laser axis) — assumes symmetric domain."""
    return g['nx'] // 2

def gmax(files, field, comp=None, pct=99.5):
    v = 0.0
    for f in files:
        a, _ = read_field(f, field, comp)
        v = max(v, float(np.percentile(np.abs(a), pct)))
    return v if v > 0 else 1.0

def gmin(files, field, comp=None):
    return min(float(read_field(f, field, comp)[0].min()) for f in files)

def make_mp4(frame_dir, pattern, out_mp4, fps=5):
    subprocess.run(
        ['ffmpeg', '-y', '-framerate', str(fps),
         '-pattern_type', 'glob', '-i', f'{frame_dir}/{pattern}',
         '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_mp4],
        check=True, capture_output=True)
    print(f'  -> {out_mp4}')


# ─────────────────────────────────────────────────────────────────────────────
# 1. LWFA  (laser Ey | wake Ez | rho) × (colormap + line at x=0)
# ─────────────────────────────────────────────────────────────────────────────
ZOOM_BEHIND, ZOOM_AHEAD, ZOOM_X = 62, 6, 20  # show full domain behind laser (domain is 68µm wide)

def make_lwfa_anim(label, diag_path, out_name, fps=4):
    files = sorted(glob.glob(f'{diag_path}/openpmd_*.json'))
    t_last = None
    if files:
        import json as _json
        with open(files[-1]) as _f: _d = _json.load(_f)
        t_last = list(_d['data'].values())[0]['attributes']['time']['value']*1e15
    print(f'  {len(files)} frames, t_max={t_last:.0f} fs')

    Ey_vm = gmax(files, 'E', 'y')
    Ez_vm = gmax(files, 'E', 'z')
    rho_min = gmin(files, 'rho')

    fdir = f'{OUT}/frames_{out_name}'
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)

    for fp in files:
        Ey, g  = read_field(fp, 'E', 'y')
        Ez, _  = read_field(fp, 'E', 'z')
        rho, _ = read_field(fp, 'rho')
        s, t   = g['step'], g['time']*1e15
        ext    = extent_um(g)
        zv     = z_axis(g)
        ix     = ix0(g)
        z_hi   = g['zmax']*1e6
        zoom_z = [z_hi - ZOOM_BEHIND, z_hi + ZOOM_AHEAD]
        zoom_x = [-ZOOM_X, ZOOM_X]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                                 gridspec_kw={'height_ratios': [2, 1]})

        for ax, data, vm, cm, lbl in [
                (axes[0,0], Ey,  Ey_vm,  'RdBu_r', 'Ey (V/m)'),
                (axes[0,1], Ez,  Ez_vm,  'RdBu_r', 'Ez (V/m)'),
                (axes[0,2], rho, abs(rho_min), 'inferno_r', 'ρ (C/m³)')]:
            vmin = -vm if cm == 'RdBu_r' else rho_min
            vmax =  vm if cm == 'RdBu_r' else 0
            im = ax.imshow(data.T, origin='lower', aspect='equal',
                           extent=ext, cmap=cm, vmin=vmin, vmax=vmax)
            ax.set_xlim(zoom_z); ax.set_ylim(zoom_x)
            plt.colorbar(im, ax=ax, label=lbl, shrink=0.8)
            ax.set_xlabel('z [µm]'); ax.set_ylabel('x [µm]')

        axes[0,0].set_title('Ey — laser (y-pol)')
        axes[0,1].set_title('Ez — accelerating wakefield')
        axes[0,2].set_title('ρ — electron density')

        iz_lo = max(0, np.searchsorted(zv, zoom_z[0]))
        iz_hi = min(g['nz'], np.searchsorted(zv, zoom_z[1]))
        zv_z  = zv[iz_lo:iz_hi]

        for col, (data, color, ylabel, title) in enumerate([
                (Ey,  'b', 'Ey (V/m)',  'Ey at x=0'),
                (Ez,  'r', 'Ez (V/m)',  'Ez at x=0'),
                (rho, 'g', 'ρ (C/m³)', 'ρ at x=0')]):
            ax = axes[1, col]
            ax.plot(zv_z, data[iz_lo:iz_hi, ix], color=color, lw=1)
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_xlabel('z [µm]'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.set_xlim(zoom_z)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1e}'))

        fig.suptitle(f'{label}  —  step {s}  |  t = {t:.0f} fs', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{fdir}/{out_name}_{s:06d}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'    step {s}  t={t:.0f} fs')

    make_mp4(fdir, f'{out_name}_*.png', f'{OUT}/{out_name}.mp4', fps=fps)


print('=== 1a. Ti:Sa LWFA — clean (AMR off, no test beam, 22 plasma periods) ===')
make_lwfa_anim(
    'Ti:Sa LWFA  [AMR off, no beam, a0=4, bubble regime]',
    f'{WORK}/tisa_lwfa_clean/diags/diag1',
    'lwfa_clean'
)

print('\n=== 1b. Ti:Sa LWFA — AMR on (exact example, dt=0.22fs, t_max=440fs) ===')
make_lwfa_anim(
    'Ti:Sa LWFA  [AMR on, dt=0.22fs]',
    f'{WORK}/tisa_lwfa_amr/diags/diag1',
    'lwfa_amr'
)

print('\n=== 1c. Ti:Sa LWFA — AMR off (dt=0.44fs, t_max=877fs) ===')
make_lwfa_anim(
    'Ti:Sa LWFA  [AMR off, dt=0.44fs]',
    f'{WORK}/tisa_lwfa_noamr/diags/diag1',
    'lwfa_noamr'
)



# ─────────────────────────────────────────────────────────────────────────────
# 2. Hole Boring  (rho_H | rho_e) × (colormap + line at x=0)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 2. Ti:Sa Hole Boring ===')
files = sorted(glob.glob(f'{WORK}/tisa_hole_boring/diags/diagInst/openpmd_*.json'))
print(f'  {len(files)} frames')

rho_h_vm  = gmax(files, 'rho_hydrogen')
rho_e_min = gmin(files, 'rho_electrons')

fdir = f'{OUT}/frames_hole_boring'
shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)

for fp in files:
    rho_h, g = read_field(fp, 'rho_hydrogen')
    rho_e, _ = read_field(fp, 'rho_electrons')
    s, t = g['step'], g['time']*1e15
    ext  = extent_um(g)
    zv   = z_axis(g)
    ix   = ix0(g)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    im0 = axes[0,0].imshow(rho_h.T, origin='lower', aspect='equal',
                            extent=ext, cmap='hot', vmin=0, vmax=rho_h_vm)
    plt.colorbar(im0, ax=axes[0,0], label='C/m³', shrink=0.8)
    axes[0,0].set_xlabel('z [µm]'); axes[0,0].set_ylabel('x [µm]')
    axes[0,0].set_title('ρ_H⁺ (hydrogen ions) — hole boring')

    im1 = axes[0,1].imshow(rho_e.T, origin='lower', aspect='equal',
                            extent=ext, cmap='inferno_r', vmin=rho_e_min, vmax=0)
    plt.colorbar(im1, ax=axes[0,1], label='C/m³', shrink=0.8)
    axes[0,1].set_xlabel('z [µm]'); axes[0,1].set_ylabel('x [µm]')
    axes[0,1].set_title('ρ_e (electrons)')

    # line profiles
    ax = axes[1,0]
    ax.plot(zv, rho_h[:, ix], color='orangered', lw=1.2, label='ρ_H⁺')
    ax.set_xlabel('z [µm]'); ax.set_ylabel('C/m³')
    ax.set_title('ρ_H⁺ at x=0')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1e}'))

    ax = axes[1,1]
    ax.plot(zv, rho_e[:, ix], color='steelblue', lw=1.2, label='ρ_e')
    ax.set_xlabel('z [µm]'); ax.set_ylabel('C/m³')
    ax.set_title('ρ_e at x=0')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1e}'))

    fig.suptitle(f'Ti:Sa Hole Boring (+z)  —  step {s}  |  t = {t:.0f} fs', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fdir}/hole_boring_{s:06d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'    step {s}  t={t:.0f} fs')

make_mp4(fdir, 'hole_boring_*.png', f'{OUT}/hole_boring.mp4', fps=3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transparency  (Ey | rho) × (colormap + line at x=0)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 3. Ti:Sa Transparency ===')
files = sorted(glob.glob(f'{WORK}/tisa_transparency/diags/diag1/openpmd_*.json'))
print(f'  {len(files)} frames')

Ey_vm_tr  = gmax(files, 'E', 'y')
rho_min_tr = gmin(files, 'rho')

fdir = f'{OUT}/frames_transparency'
shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)

for fp in files:
    Ey, g  = read_field(fp, 'E', 'y')
    rho, _ = read_field(fp, 'rho')
    s, t   = g['step'], g['time']*1e15
    ext    = extent_um(g)
    zv     = z_axis(g)
    ix     = ix0(g)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    im0 = axes[0,0].imshow(Ey.T, origin='lower', aspect='equal',
                            extent=ext, cmap='RdBu_r',
                            vmin=-Ey_vm_tr, vmax=Ey_vm_tr)
    plt.colorbar(im0, ax=axes[0,0], label='V/m', shrink=0.8)
    axes[0,0].set_xlabel('z [µm]'); axes[0,0].set_ylabel('x [µm]')
    axes[0,0].set_title('Ey — laser field (a₀≈1, reflecting)')

    im1 = axes[0,1].imshow(rho.T, origin='lower', aspect='equal',
                            extent=ext, cmap='inferno_r',
                            vmin=rho_min_tr, vmax=0)
    plt.colorbar(im1, ax=axes[0,1], label='C/m³', shrink=0.8)
    axes[0,1].set_xlabel('z [µm]'); axes[0,1].set_ylabel('x [µm]')
    axes[0,1].set_title('ρ — charge density')

    ax = axes[1,0]
    ax.plot(zv, Ey[:, ix], 'b-', lw=1)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('z [µm]'); ax.set_ylabel('Ey (V/m)')
    ax.set_title('Ey at x=0 — incident + reflected laser')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1e}'))

    ax = axes[1,1]
    ax.plot(zv, rho[:, ix], 'g-', lw=1)
    ax.set_xlabel('z [µm]'); ax.set_ylabel('ρ (C/m³)')
    ax.set_title('ρ at x=0 — plasma density profile')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1e}'))

    fig.suptitle(f'Ti:Sa Transparency (+z)  —  step {s}  |  t = {t:.0f} fs', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fdir}/transparency_{s:06d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'    step {s}  t={t:.0f} fs')

make_mp4(fdir, 'transparency_*.png', f'{OUT}/transparency.mp4', fps=5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. KrF Laser-Plasma  (Ex | rho) × (colormap + line at x=0)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== 4. KrF Laser-Plasma ===')
files = sorted(glob.glob(f'{WORK}/krf_laser_plasma/diags/diag1/openpmd_*.json'))
print(f'  {len(files)} frames')

Ex_vm_krf  = gmax(files, 'E', 'x')
rho_min_krf = gmin(files, 'rho')

fdir = f'{OUT}/frames_krf'
shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)

for fp in files:
    Ex, g  = read_field(fp, 'E', 'x')
    rho, _ = read_field(fp, 'rho')
    s, t   = g['step'], g['time']*1e15
    ext    = extent_um(g)
    zv     = z_axis(g)
    ix     = ix0(g)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    im0 = axes[0,0].imshow(Ex.T, origin='lower', aspect='equal',
                            extent=ext, cmap='RdBu_r',
                            vmin=-Ex_vm_krf, vmax=Ex_vm_krf)
    plt.colorbar(im0, ax=axes[0,0], label='V/m', shrink=0.8)
    axes[0,0].set_xlabel('z [µm]'); axes[0,0].set_ylabel('x [µm]')
    axes[0,0].set_title('Ex — KrF laser (248 nm, x-pol, a₀≈2)')

    im1 = axes[0,1].imshow(rho.T, origin='lower', aspect='equal',
                            extent=ext, cmap='inferno_r',
                            vmin=rho_min_krf, vmax=0)
    plt.colorbar(im1, ax=axes[0,1], label='C/m³', shrink=0.8)
    axes[0,1].set_xlabel('z [µm]'); axes[0,1].set_ylabel('x [µm]')
    axes[0,1].set_title('ρ — charge density')

    ax = axes[1,0]
    ax.plot(zv, Ex[:, ix], 'b-', lw=0.8)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('z [µm]'); ax.set_ylabel('Ex (V/m)')
    ax.set_title('Ex at x=0 — laser + reflected')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1e}'))

    ax = axes[1,1]
    ax.plot(zv, rho[:, ix], 'g-', lw=0.8)
    ax.set_xlabel('z [µm]'); ax.set_ylabel('ρ (C/m³)')
    ax.set_title('ρ at x=0 — density profile + hole')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1e}'))

    fig.suptitle(f'KrF Laser-Plasma (+z)  —  step {s}  |  t = {t:.0f} fs', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fdir}/krf_{s:06d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'    step {s}  t={t:.0f} fs')

make_mp4(fdir, 'krf_*.png', f'{OUT}/krf.mp4', fps=5)

print('\n=== All done ===')
for f in sorted(glob.glob(f'{OUT}/*.mp4')):
    print(f'  {os.path.basename(f)}  ({os.path.getsize(f)//1024} KB)')
