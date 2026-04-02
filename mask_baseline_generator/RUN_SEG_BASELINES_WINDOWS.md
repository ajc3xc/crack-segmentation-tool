# Windows Baseline Export (Pixi)

Input images:
`C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Original_Image`

Output root:
`C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs\seg_baselines`

Result layout:
`<output_root>\hrsegnet\*.png` and `<output_root>\deeplab\*.png`

All exported masks are saved at original image resolution with values `0` (non-crack) and `255` (crack).

## 1) HrSegNet (b32, default)

From repo root (`mask_baseline_generator`):

```powershell
pixi run python .\hrsegnet\HrSegNet4CrackSegmentation\export_hrsegnet_baseline_masks.py `
  --variant b32 `
  --method-name hrsegnet `
  --output-root "C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs\seg_baselines"
```

Optional b16 run (extra comparison):

```powershell
pixi run python .\hrsegnet\HrSegNet4CrackSegmentation\export_hrsegnet_baseline_masks.py `
  --variant b16 `
  --method-name hrsegnet_b16 `
  --output-root "C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs\seg_baselines"
```

## 2) DeepLab Xception

From `pytorch_deeplab_xception` folder:

```powershell
pixi run python .\export_deeplab_baseline_masks.py `
  --weights "C:\path\to\your\deeplab\checkpoint.pth.tar" `
  --method-name deeplab `
  --output-root "C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs\seg_baselines"
```

If needed, force CPU:

```powershell
pixi run python .\export_deeplab_baseline_masks.py `
  --weights "C:\path\to\your\deeplab\checkpoint.pth.tar" `
  --device cpu
```
