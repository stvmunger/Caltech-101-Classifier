import os
import random

def split_dataset(root_dir, train_list_path, test_list_path, num_train_per_class):
    """
    将 root_dir 里的每个类随机抽取 num_train_per_class 张图做 train，
    其余做 test，输出两份 txt，每行：<绝对路径> <类别索引>
    """
    classes = sorted([
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
       and d != "BACKGROUND_Google"      # ← 忽略背景类
])
    
    cls2idx = {cls: idx for idx, cls in enumerate(classes)}

    with open(train_list_path, 'w') as f_train, \
         open(test_list_path,  'w') as f_test:

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            imgs = sorted([
                fn for fn in os.listdir(cls_dir)
                if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            random.seed(42)
            random.shuffle(imgs)

            train_imgs = imgs[:num_train_per_class]
            test_imgs  = imgs[num_train_per_class:]

            for fn in train_imgs:
                path = os.path.join(cls_dir, fn)
                f_train.write(f"{path} {cls2idx[cls]}\n")
            for fn in test_imgs:
                path = os.path.join(cls_dir, fn)
                f_test.write(f"{path} {cls2idx[cls]}\n")

    print(f"Saved train list → {train_list_path}")
    print(f"Saved test  list → {test_list_path}")
