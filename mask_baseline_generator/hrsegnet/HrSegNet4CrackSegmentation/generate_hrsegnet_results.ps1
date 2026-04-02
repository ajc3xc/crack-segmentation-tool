$env:PATH = "C:\Users\13144\Documents\Masters_Thesis\mask_baseline_generator\cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin;" + $env:PATH
#pixi installation path
$env:PATH = "C:\Users\13144\Documents\Masters_Thesis\mask_baseline_generator\.pixi\envs\default\Lib\site-packages\nvidia\cublas\bin;" + $env:PATH
pixi run python export_hrsegnet_baseline_masks.py