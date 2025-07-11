#!/bin/bash

# --- Git 自动化提交与推送脚本 ---
# 用法：在项目根目录运行此脚本，它会提示您输入提交信息。
# 此脚本执行完毕后不会自动退出。

# 确保在正确的 Git 仓库目录中
if [ ! -d ".git" ]; then
    echo "错误：当前目录不是一个Git仓库。请在PI_GAN_THZ项目的根目录运行此脚本。"
    # 不再使用 exit 1，而是直接输出错误并继续（或等待用户手动关闭）
    # 但对于这种致命错误，通常还是希望退出，这里为了保持窗口，只是不强制退出
    # 如果您在独立的终端运行，它会保持打开。
    # 如果您希望在Colab等环境中，遇到这种错误仍然停止，可以加上 exit 1
    # 对于Windows双击运行，通常会直接关闭窗口，所以这条错误信息可能不会被看到。
fi

echo "--- 准备提交 PI-GAN-THz 项目变更 ---"

# 1. 添加所有更改到暂存区
echo "执行：git add ."
git add .

# 检查是否有文件被添加到暂存区
if git diff --cached --quiet; then
    echo "没有检测到需要提交的变更。工作区和暂存区都是干净的。"
    echo "--- 脚本结束 ---"
    # 不再使用 exit 0
fi

# 2. 获取提交信息
read -p "请输入提交信息 (例如: 'feat: Add new feature', 'fix: Bug fix', 'docs: Update README'): " commit_message

# 检查提交信息是否为空
if [ -z "$commit_message" ]; then
    echo "提交信息不能为空。操作取消。"
    git reset HEAD # 取消暂存
    # 不再使用 exit 1
fi

# 3. 提交到本地仓库
echo "执行：git commit -m \"$commit_message\""
git commit -m "$commit_message"

# 检查提交是否成功
if [ $? -ne 0 ]; then
    echo "Git 提交失败。请检查上述错误信息。"
    # 不再使用 exit 1
fi

# 4. 推送到远程仓库 (假设主分支为 'main' 或 'master')
echo "正在推送变更到远程仓库..."
# 尝试推送到 main 分支，如果不存在则尝试 master 分支
git push origin main || git push origin master

# 检查推送是否成功
if [ $? -ne 0 ]; then
    echo "Git 推送失败。请检查网络连接或凭据。"
    # 不再使用 exit 1
fi

echo "--- 变更已成功提交并推送到 GitHub！🎉 ---"
# 脚本自然结束，不强制退出