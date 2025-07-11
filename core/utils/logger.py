# PI_GAN_THZ/core/utils/logger.py

import logging
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

class Logger:
    """
    统一的日志管理类，支持控制台输出、文件记录和 TensorBoard。
    """
    def __init__(self, log_dir: str, experiment_name: str = "default_experiment"):
        """
        初始化 Logger。

        Args:
            log_dir (str): 日志文件和 TensorBoard 日志保存的根目录。
            experiment_name (str): 当前实验的名称，用于创建独立的日志子目录。
        """
        # 创建当前实验的日志目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_log_dir, exist_ok=True)

        # --- 设置 Python 内置日志 ---
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO) # 设置最低日志级别为 INFO

        # 避免重复添加处理器
        if not self.logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # 文件处理器
            log_file_path = os.path.join(self.experiment_log_dir, f"{experiment_name}.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file_path}")

        # --- 设置 TensorBoard SummaryWriter ---
        self.writer = SummaryWriter(log_dir=self.experiment_log_dir)
        self.logger.info(f"TensorBoard logs will be saved to: {self.experiment_log_dir}")

    def info(self, message: str):
        """记录 INFO 级别的日志信息。"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录 WARNING 级别的日志信息。"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录 ERROR 级别的日志信息。"""
        self.logger.error(message)

    def add_scalar(self, tag: str, scalar_value, global_step: int):
        """
        向 TensorBoard 记录一个标量值。

        Args:
            tag (str): 数据的标签 (例如 'Loss/Generator_Loss')。
            scalar_value: 要记录的标量值。
            global_step (int): 当前的训练步数或 epoch 数。
        """
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int):
        """
        向 TensorBoard 记录一组标量值 (在同一张图中显示)。

        Args:
            main_tag (str): 主标签 (例如 'Losses')。
            tag_scalar_dict (dict): 包含多个子标签和对应值的字典。
            global_step (int): 当前的训练步数或 epoch 数。
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    # 可以根据需要添加更多 TensorBoard 方法，如 add_histogram, add_image 等

    def close(self):
        """关闭 TensorBoard SummaryWriter。"""
        self.writer.close()
        self.logger.info("Logger closed.")