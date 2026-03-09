import os

def rename_images(folder_path):
    """
    重命名指定文件夹下所有图片文件。
    将文件名从类似 img001x4_test_SSANet_x4_light.png 改为 img_001x4_test_SSANet_x4_light.png，
    即在 img 后添加下划线 _。
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在！")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以.png结尾
        if filename.endswith(".png"):
            # 检查文件名是否以 img 开头
            if filename.startswith("img"):
                # 在 img 后添加下划线 _
                new_filename = filename.replace("img_", "img", 1)
            else:
                # 如果文件名不是以 img 开头，跳过
                print(f"文件 {filename} 不是以 'img' 开头，跳过重命名")
                continue

            # 构造完整的旧文件路径和新文件路径
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            # 检查新文件名是否已存在，避免覆盖
            if os.path.exists(new_file_path):
                print(f"文件 {new_filename} 已存在，跳过重命名 {filename}")
            else:
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"文件 {filename} 已重命名为 {new_filename}")

# 使用示例
folder_path = "results/test_SSANet_x4/visualization/Urban100"  # 替换为你的文件夹路径
rename_images(folder_path)