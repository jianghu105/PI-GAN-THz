#!/bin/bash

# --- Git 拉取并合并代码脚本 ---
# 用法：在项目根目录运行此脚本，它会提示您选择拉取的分支。

# 确保在正确的 Git 仓库目录中
if [ ! -d ".git" ]; then
    echo "错误：当前目录不是一个Git仓库。请在PI_GAN_THZ项目的根目录运行此脚本。"
    # 遇到致命错误时，通常还是希望退出，这里为了保持窗口，只是不强制退出
    # 如果您希望在Colab等环境中，遇到这种错误仍然停止，可以加上 exit 1
fi

echo "--- 准备从远程仓库拉取并合并代码 ---"

# 获取当前分支
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "当前所在分支：$current_branch"

# 询问用户要拉取哪个分支
read -p "请输入要拉取的分支名称（留空则默认拉取当前分支 '$current_branch'）： " branch_to_pull

# 如果用户没有输入，则默认拉取当前分支
if [ -z "$branch_to_pull" ]; then
    branch_to_pull="$current_branch"
    echo "将从远程仓库拉取并合并当前分支：$current_branch"
else
    echo "将从远程仓库拉取并合并分支：$branch_to_pull"
fi

# 执行 git pull
echo "执行：git pull origin $branch_to_pull"
git pull origin "$branch_to_pull"

# 检查拉取是否成功
if [ $? -ne 0 ]; then
    echo "Git 拉取失败。请检查网络连接、分支名称或合并冲突。"
    echo "如果存在合并冲突，您需要手动解决它们，然后提交并推送。"
fi

echo "--- 代码拉取并合并操作完成 ---"