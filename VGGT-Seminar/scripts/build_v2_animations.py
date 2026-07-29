"""Create deterministic point-cloud renders and animations from saved ASCII PLY files only."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/v2/animations"
FIG = ROOT / "report/v2/figures"
BG = (12, 18, 28)
W, H, FPS, FRAMES = 960, 640, 10, 80

PLY = {
    "delivery_area": ROOT / "outputs/predictions/phase7_eth3d_view_count/delivery_area/S10_overlap_aware_nested_original/visualizations/point_cloud_confidence_filtered.ply",
    "courtyard": ROOT / "outputs/predictions/phase8_eth3d_view_count/courtyard/S10_overlap_aware_nested_original/visualizations/point_cloud_confidence_filtered.ply",
}

def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try: return ImageFont.truetype(name, size)
    except OSError: return ImageFont.load_default()

def load_sample(path, target=45000):
    with path.open(encoding="ascii") as f:
        count = 0
        for line in f:
            if line.startswith("element vertex"): count = int(line.split()[-1])
            if line.strip() == "end_header": break
        stride = max(1, count // target)
        xyz, rgb = [], []
        for i, line in enumerate(f):
            if i % stride: continue
            values = line.split()
            if len(values) >= 6:
                xyz.append([float(v) for v in values[:3]])
                rgb.append([int(v) for v in values[3:6]])
    xyz = np.asarray(xyz, np.float32); rgb = np.asarray(rgb, np.uint8)
    center = np.median(xyz, axis=0); xyz -= center
    radius = np.percentile(np.linalg.norm(xyz, axis=1), 97)
    keep = np.linalg.norm(xyz, axis=1) <= radius
    return xyz[keep] / max(radius, 1e-6), rgb[keep]

def render(points, colors, angle, label):
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]], np.float32)
    p = points @ rot.T
    tilt = np.deg2rad(-12)
    rt = np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]],np.float32)
    p = p @ rt.T
    z = p[:,2] + 3.0
    x = W/2 + 430 * p[:,0] / z
    y = H/2 - 430 * p[:,1] / z
    valid=(x>=8)&(x<W-8)&(y>=58)&(y<H-32)&(z>0)
    order=np.argsort(z[valid])[::-1]; x=x[valid][order].astype(int); y=y[valid][order].astype(int); c=colors[valid][order]
    arr=np.empty((H,W,3),np.uint8); arr[:]=BG
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)): arr[np.clip(y+dy,0,H-1),np.clip(x+dx,0,W-1)]=c
    im=Image.fromarray(arr); d=ImageDraw.Draw(im)
    d.rounded_rectangle((22,18,330,54),radius=12,fill=(22,33,49),outline=(65,85,110),width=1)
    d.text((38,27),f"{label.replace('_',' ')} | S10 | saved PLY",font=font(18,True),fill=(238,244,252))
    d.text((25,H-25),"Deterministic perspective render - arbitrary unaligned prediction units",font=font(14),fill=(159,176,197))
    return im

def build(scene, path):
    points, colors=load_sample(path)
    frames=[render(points,colors,i*360/FRAMES,scene) for i in range(FRAMES)]
    frames[0].save(OUT/f"{scene}_s10_rotation.gif",save_all=True,append_images=frames[1:],duration=1000//FPS,loop=0,optimize=True)
    writer=cv2.VideoWriter(str(OUT/f"{scene}_s10_rotation.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),FPS,(W,H))
    for im in frames: writer.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR))
    writer.release()
    key=[frames[i] for i in (0,16,32,48,64)]
    strip=Image.new("RGB",(W,len(key)*(H//2)),(245,247,250))
    for i,im in enumerate(key): strip.paste(im.resize((W,H//2),Image.Resampling.LANCZOS),(0,i*(H//2)))
    strip.save(FIG/f"{scene}_animation_keyframes.png")
    frames[0].save(FIG/f"{scene}_s10_pointcloud_hero.png")

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    for scene,path in PLY.items():
        if not path.is_file(): raise FileNotFoundError(path)
        build(scene,path)
    print({"scenes":2,"frames_each":FRAMES,"duration_seconds":FRAMES/FPS})

if __name__ == "__main__": main()
