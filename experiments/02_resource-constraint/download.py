# download_models.py
import os
from transformers import (
    AutoImageProcessor,
    Swin2SRForImageSuperResolution,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    DetrImageProcessor,
    DetrForObjectDetection
)

# 定义所有模型的名称和它们对应的加载器
MODEL_LIST = {
    # 超分模型
    "caidas/swin2SR-classical-sr-x2-64": [Swin2SRForImageSuperResolution, AutoImageProcessor],
}

# 指定一个本地目录来存放所有模型文件
SAVE_DIRECTORY = "./models"

def download_all_models():
    """下载所有指定的模型到本地目录"""
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        print(f"Created directory: {SAVE_DIRECTORY}")

    for model_name, loaders in MODEL_LIST.items():
        print(f"--- Downloading {model_name} ---")
        # 为每个模型创建一个子目录
        model_path = os.path.join(SAVE_DIRECTORY, model_name)

        for loader_class in loaders:
            try:
                # 加载并保存
                loader_instance = loader_class.from_pretrained(model_name)
                loader_instance.save_pretrained(model_path)
                print(f"Successfully downloaded and saved {loader_class.__name__}")
            except Exception as e:
                print(f"Error downloading {loader_class.__name__} for {model_name}: {e}")
        print(f"--- Finished {model_name} ---\n")

if __name__ == "__main__":
    download_all_models()
