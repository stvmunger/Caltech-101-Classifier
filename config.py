import os

ROOT_DIR      = os.path.dirname(__file__)
DATA_DIR      = os.path.join(ROOT_DIR, 'data', '101_ObjectCategories')
SCRIPTS_DIR   = os.path.join(ROOT_DIR, 'scripts')
TRAIN_LIST    = os.path.join(SCRIPTS_DIR, 'train.txt')
TEST_LIST     = os.path.join(SCRIPTS_DIR, 'test.txt')
LOG_DIR       = os.path.join(ROOT_DIR, 'logs')
MODEL_DIR     = os.path.join(ROOT_DIR, 'models')

# 确保关键目录存在
for d in [DATA_DIR, SCRIPTS_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

NUM_CLASSES   = 101
TRAIN_SAMPLES = 30   # 每类用于训练的样本数

# 默认超参，可由 CLI 覆盖
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LR_BACKBONE   = 1e-4
LR_HEAD       = 1e-3
WEIGHT_DECAY  = 1e-4
SAVE_FREQ     = 5    # 每 SAVE_FREQ 个 epoch 保存一次

# 设备自动选择
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
