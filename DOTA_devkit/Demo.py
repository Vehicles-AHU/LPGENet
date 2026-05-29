import os

labels_folder = "./datasets_vedai/labels"
output_file = "./filename_vedai.txt"


def write_filenames_sorted_numeric(labels_folder, output_file):
    txt_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    # 按数字顺序排序
    txt_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in os.path.splitext(x)[0]])
    
    filenames = [os.path.splitext(f)[0] for f in txt_files]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in filenames:
            f.write(filename + '\n')
    
    print(f"成功将 {len(filenames)} 个文件名按数字顺序写入到 {output_file}")
    
write_filenames_sorted_numeric(labels_folder, "./filenames_sorted.txt")