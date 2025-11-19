import re
import json
import sys
import os  # 导入 os 模块来处理路径


def parse_detailed_log(log_path):
    """
    解析一个详细的、多轮(multi-epoch)的训练日志文件。
    找到验证集(validation)上 '***auc' 最高的那个 epoch，并返回其所有指标。
    """

    # 正则表达式，用于匹配关键信息行
    # 匹配开始训练的 epoch 行，例如: ... [INFO] Start training epoch 0 ...
    epoch_regex = re.compile(r"Start training epoch (\d+)")

    # 匹配验证集的指标行，例如:
    # ... [INFO] Averaged stats: loss: 0.632... acc: 0.651... ***auc: 0.704... ***uauc: 0.659... ***u-nDCG: 0.854...
    #
    # 更新：在正则表达式中加入了 [INFO] + 前缀，使其匹配更精确
    metrics_regex = re.compile(
        r"\[INFO\] +Averaged stats: loss: (\S+) +acc: (\S+) +\*\*\*auc: (\S+) +\*\*\*uauc: (\S+) +\*\*\*u-nDCG: (\S+)"
    )

    best_result_metrics = None
    best_uauc = -1.0
    best_auc = -1.0
    current_epoch = -1

    # 我们需要一个标志来确保我们只读取评估(Evaluating)阶段的 "Averaged stats"
    # 而不是训练(training)阶段的 "Averaged stats"
    # is_in_evaluation_block = False # 移除此行

    try:
        # 修复：将编码从 'utf-8' 更改为 'gbk'，以解决中文 Windows 环境下的解码错误
        with open(log_path, 'r', encoding='gbk') as f:
            for line in f:
                # 1. 检查是否开始了新的 epoch
                epoch_match = epoch_regex.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    # 开始了新的训练轮次，重置评估标志
                    # is_in_evaluation_block = False # 移除此行

                # 2. 检查是否进入了验证阶段 # 移除此逻辑块
                # if "[INFO] Evaluating on valid." in line:
                #     is_in_evaluation_block = True

                # 3. 检查是否是指标行
                # 必须在评估阶段 (is_in_evaluation_block == True)
                # 并且 匹配 metrics_regex
                metrics_match = metrics_regex.search(line)
                if metrics_match:  # 移除了 is_in_evaluation_block 检查
                    try:
                        # 提取所有指标
                        valid_loss = float(metrics_match.group(1))
                        valid_acc = float(metrics_match.group(2))
                        valid_auc = float(metrics_match.group(3))
                        valid_uauc = float(metrics_match.group(4))
                        valid_unDCG = float(metrics_match.group(5))

                        current_result = {
                            "epoch": current_epoch,
                            "valid_loss": valid_loss,
                            "valid_acc": valid_acc,
                            "valid_auc": valid_auc,
                            "valid_uauc": valid_uauc,
                            "valid_unDCG": valid_unDCG
                        }

                        # 4. 检查这是否是目前最好的结果
                        # if valid_uauc > best_uauc:
                        #     best_uauc = valid_uauc
                        #     best_result_metrics = current_result

                        if valid_auc > best_auc:
                            best_auc = valid_auc
                            best_result_metrics = current_result

                        # 找到指标后，重置评估标志，防止重复读取
                        # is_in_evaluation_block = False # 移除此行

                    except ValueError as e:
                        print(f"警告: 无法解析指标行: {line.strip()}", file=sys.stderr)
                        print(f"错误详情: {e}", file=sys.stderr)

    except FileNotFoundError:
        print(f"错误: 找不到日志文件: {log_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"解析日志时发生错误: {e}", file=sys.stderr)
        return None

    return best_result_metrics


# --- 脚本主执行区 ---
if __name__ == "__main__":

    # ---! 请在这里修改你的日志文件名 !---
    #
    # 确保这个文件是你的详细训练日志 (例如 W1114 08:55:34... 那个)
    # 并且它和 parse_run_log.py 脚本在同一个文件夹下
    #
    log_filename = "1outcome/1113_LoRA-ppllm-qwen-mf-ml.txt"
    # log_filename = "1outcome/1027_LoRA-tallrec-qwen-mf-ml_new.txt"
    # log_filename = "1113_evaluate_LoRA-ppllm-qwen-mf-ml.txt"
    # log_filename = "1113_evaluate_LoRA-ppllm-qwen-mf-ml_oldcheckpoint.txt"
    # log_filename = "evaluate/1027_evaluate_LoRA_tallrec_qwen_mf_ml.txt"
    #
    # ---! 修改结束 !---

    # --- 自动构建路径 ---
    # __file__ 是当前脚本 (parse_run_log.py) 的路径
    # os.path.dirname(__file__) 是脚本所在的文件夹
    # os.path.abspath() 确保我们得到一个绝对路径
    try:
        script_dir = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        # 如果在某些环境 (如 REPL) 中 __file__ 未定义，则使用当前工作目录
        script_dir = os.path.abspath(os.getcwd())

    # 将脚本所在目录与日志文件名组合成一个绝对路径
    log_file_path = os.path.join(script_dir, log_filename)

    # 打印出脚本正在尝试读取的完整路径，方便调试
    print(f"正在尝试读取日志文件: {log_file_path}")

    best_stats = parse_detailed_log(log_file_path)

    if best_stats:
        print(f"日志分析完成: {log_file_path}")
        print("\n--- 最佳 Epoch 结果 (基于最高 valid_auc) ---")

        # 使用 json.dumps 格式化输出，更易读
        print(json.dumps(best_stats, indent=4))

        print("\n--- (可选) 如何与 find_max.py 配合使用 ---")
        print("你可以手动将此结果格式化为 '总结日志' 的一行，例如：")
        # 我们可以创建一个虚拟的 config，因为解析它很复杂
        dummy_config = {"log_file": log_file_path, "note": "result_from_parse_run_log.py"}
        print(f"\ntrain_config: {dummy_config} best result: {best_stats}")

    else:
        print(f"错误: 未能在 {log_file_path} 中找到任何有效的验证指标。")
        print("请检查：")
        print("  1. 文件路径是否正确。")
        print("  2. 日志文件是否包含 `[INFO] Averaged stats: ... ***auc: ...` 行。")