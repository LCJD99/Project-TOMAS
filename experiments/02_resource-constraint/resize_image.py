import os
from PIL import Image
import argparse

def resize_image(image_path, width, height):
    """
    将图片调整到指定的宽度和高度。

    Args:
        image_path (str): 原始图片的路径。
        width (int): 目标宽度。
        height (int): 目标高度。

    Returns:
        str: 新图片的保存路径，如果失败则返回 None。
    """
    if not os.path.exists(image_path):
        print(f"错误：文件 '{image_path}' 不存在。")
        return None

    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 调整图片大小
            # ANTIALIAS 参数可以在缩放时保持较高的图像质量
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)

            # 创建新的文件名
            file_name, file_ext = os.path.splitext(image_path)
            new_image_path = f"{file_name}_resized{file_ext}"

            # 保存调整大小后的图片
            resized_img.save(new_image_path)
            print(f"图片已成功调整大小并保存为 '{new_image_path}'")
            return new_image_path

    except Exception as e:
        print(f"处理图片时发生错误: {e}")
        return None

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="将图片压缩到指定的宽度和高度。")

    # 添加参数
    parser.add_argument("image_path", type=str, help="要处理的原始图片路径。")
    parser.add_argument("width", type=int, help="压缩后的目标宽度（像素）。")
    parser.add_argument("height", type=int, help="压缩后的目标高度（像素）。")

    # 解析参数
    args = parser.parse_args()

    # 调用函数
    resize_image(args.image_path, args.width, args.height)
