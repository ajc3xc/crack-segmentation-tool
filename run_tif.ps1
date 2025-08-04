$env:PATH = "D:\camerer_ml\cudnn\cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin;" + $env:PATH
$env:PATH = "D:\camerer_ml\.pixi\envs\default\Lib\site-packages\nvidia\cublas\bin;" + $env:PATH
pixi run python tif_raster_overlay.py