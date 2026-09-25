$ErrorActionPreference = "Stop"
$env:TRAFFIC_CONFIG = "configs/data_video1_dev.json"
python run_submission.py --videos data/data_video1.mp4 --out data/inspection/data_video1_predictions.json --team dev
Remove-Item Env:TRAFFIC_CONFIG -ErrorAction SilentlyContinue
