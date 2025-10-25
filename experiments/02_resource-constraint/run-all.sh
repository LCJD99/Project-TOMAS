echo ImageCaptioning
python run.py --image_path data/lenna_resized.jpg
echo ObjectDetection
python run.py --task ObjectDetection --image_path data/lenna_resized.jpg
echo SuperResolution
python run.py  --task SuperResolution --image_path data/lenna_resized.jpg
