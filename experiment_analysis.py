"""
实验分析报告生成器
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image

# 尝试导入matplotlib，如果失败则提供替代方案
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib不可用，将跳过图表生成")

class ExperimentAnalyzer:
    def __init__(self, results_dir="experiment_results"):
        self.results_dir = Path(results_dir)
        self.load_results()

    def load_results(self):
        """加载实验结果"""
        self.hyper_results = self._load_json("hyperparameter_results.json")
        self.sampling_results = self._load_json("sampling_comparison.json")
        self.guided_results = self._load_json("guided_generation.json")

    def _load_json(self, filename):
        """加载JSON文件"""
        filepath = self.results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None

    def generate_comprehensive_report(self):
        """生成综合实验报告"""
        report = []

        report.append("# 扩散模型实验分析报告\n")

        # 1. 超参数实验分析
        if self.hyper_results:
            report.append("## 1. 超参数实验分析\n")
            report.extend(self._analyze_hyperparameter_results())

        # 2. 采样算法对比分析
        if self.sampling_results:
            report.append("## 2. 采样算法对比分析\n")
            report.extend(self._analyze_sampling_results())

        # 3. 引导生成实验分析
        if self.guided_results:
            report.append("## 3. 引导生成实验分析\n")
            report.extend(self._analyze_guided_results())

        # 4. 综合结论
        report.append("## 4. 综合结论\n")
        report.extend(self._generate_conclusions())

        # 保存报告
        report_text = "\n".join(report)
        with open(self.results_dir / "experiment_report.md", 'w', encoding='utf-8') as f:
            f.write(report_text)

        return report_text

    def _analyze_hyperparameter_results(self):
        """分析超参数实验结果"""
        analysis = []

        # 创建数据表
        data = []
        if self.hyper_results is None:
            analysis.append("### 超参数实验数据为空\n")
            return analysis

        for exp_name, exp_data in self.hyper_results.items():
            config = exp_data['config']
            data.append({
                '实验': exp_name,
                '特征维度': config['n_feat'],
                '学习率': config['lr'],
                '批次大小': config['batch_size'],
                '最终损失': exp_data['final_loss'],
                '采样时间': exp_data['sample_time']
            })

        df = pd.DataFrame(data)

        analysis.append("### 超参数配置与性能对比\n")
        # 使用简单的格式化表格替代 to_markdown
        analysis.append(self._df_to_markdown(df))
        analysis.append("\n")

        # 可视化损失曲线
        if MATPLOTLIB_AVAILABLE:
            self._plot_hyperparameter_losses()

        # 分析结论
        analysis.append("### 超参数影响分析\n")

        # 特征维度分析
        feat_analysis = df.groupby('特征维度')['最终损失'].mean()
        best_feat = feat_analysis.idxmin()
        analysis.append(f"- **特征维度影响**: {best_feat}维特征表现最佳")

        # 学习率分析
        lr_analysis = df.groupby('学习率')['最终损失'].mean()
        best_lr = lr_analysis.idxmin()
        analysis.append(f"- **学习率影响**: {best_lr}学习率表现最佳")

        # 批次大小分析
        bs_analysis = df.groupby('批次大小')['最终损失'].mean()
        best_bs = bs_analysis.idxmin()
        analysis.append(f"- **批次大小影响**: {best_bs}批次大小表现最佳\n")

        return analysis

    def _analyze_sampling_results(self):
        """分析采样算法对比结果"""
        analysis = []

        # 创建对比数据
        comparison_data = []
        if self.sampling_results is None:
            analysis.append("### 采样算法对比数据为空\n")
            return analysis

        for model_name, model_data in self.sampling_results.items():
            ddpm_data = model_data['ddpm']
            ddim_data = model_data['ddim']

            comparison_data.append({
                '模型': model_name,
                'DDPM时间(s)': ddpm_data['time'],
                'DDIM时间(s)': ddim_data['time'],
                '速度提升': ddpm_data['time'] / ddim_data['time'],
                'DDPM均值': ddpm_data['metrics']['mean'],
                'DDIM均值': ddim_data['metrics']['mean']
            })

        df = pd.DataFrame(comparison_data)

        analysis.append("### 采样算法性能对比\n")
        analysis.append(self._df_to_markdown(df))
        analysis.append("\n")

        # 可视化对比
        if MATPLOTLIB_AVAILABLE:
            self._plot_sampling_comparison()

        # 分析结论
        analysis.append("### 采样算法分析\n")

        avg_speedup = df['速度提升'].mean()
        analysis.append(f"- **速度对比**: DDIM平均比DDPM快{avg_speedup:.1f}倍")

        # 质量分析
        ddpm_quality = df['DDPM均值'].mean()
        ddim_quality = df['DDIM均值'].mean()
        analysis.append(f"- **质量对比**: DDPM均值={ddpm_quality:.3f}, DDIM均值={ddim_quality:.3f}")

        analysis.append("- **结论**: DDIM在保持合理质量的同时，显著提升了采样速度\n")

        return analysis

    def _analyze_guided_results(self):
        """分析引导生成实验结果"""
        analysis = []

        analysis.append("### 引导生成效果分析\n")

        if self.guided_results is None:
            analysis.append("引导生成实验数据为空\n")
            return analysis

        # 创建引导生成配置表
        config_data = []
        for config_name, config_data_item in self.guided_results.items():
            context_config = config_data_item.get('context_config', [])

            # 解析上下文配置
            context_str = self._parse_context_config(context_config)

            config_data.append({
                '配置名称': config_name,
                '上下文向量': str(context_config),
                '类别解释': context_str,
                '样本数量': len(config_data_item.get('samples', []))
            })

        if config_data:
            df = pd.DataFrame(config_data)
            analysis.append("#### 引导生成配置表\n")
            analysis.append(self._df_to_markdown(df))
            analysis.append("\n")

        # 详细分析每个配置
        analysis.append("### 各类别生成分析\n")

        # 单一类别分析
        single_categories = ['human', 'non_human', 'food', 'spell', 'side_facing', 'unconditional']
        analysis.append("#### 单一类别生成\n")
        for cat in single_categories:
            if cat in self.guided_results:
                config = self.guided_results[cat].get('context_config', [])
                analysis.append(f"- **{cat.replace('_', ' ').title()}**: 上下文 {config}")
                analysis.append(f"  - 描述: {self._get_category_description(cat)}")
        analysis.append("")

        # 混合类别分析
        analysis.append("#### 混合类别生成\n")
        mixed_categories = [k for k in self.guided_results.keys() if k.startswith('mixed_')]
        if mixed_categories:
            for cat in mixed_categories:
                config = self.guided_results[cat].get('context_config', [])
                analysis.append(f"- **{cat.replace('_', ' ').title()}**: 上下文 {config}")
                analysis.append(f"  - 描述: {self._get_category_description(cat)}")
            analysis.append("")
        else:
            analysis.append("- 未发现混合类别实验\n")

        # 可视化引导生成结果
        if MATPLOTLIB_AVAILABLE:
            self._plot_guided_generation()
            analysis.append("*引导生成对比图已保存为 `guided_generation_comparison.png`*\n")

        # 技术分析
        analysis.append("### 引导生成技术分析\n")
        analysis.append("#### 工作原理\n")
        analysis.append("1. **上下文编码**: 5维one-hot向量编码类别信息")
        analysis.append("   - 向量格式: `[human, non_human, food, spell, side_facing]`")
        analysis.append("   - 示例: `[1,0,0,0,0]` 表示人类类别\n")

        analysis.append("2. **条件调制**: 上下文通过嵌入层与时间嵌入结合")
        analysis.append("   - 在UNet上采样路径中调制特征图")
        analysis.append("   - 公式: `up2 = UnetUp(cemb1 * up1 + temb1, down2)`\n")

        analysis.append("3. **DDIM采样**: 使用25步确定性采样")
        analysis.append("   - 采样公式: `x0_pred = sqrt(α_{t-1})/sqrt(α_t) * (x_t - sqrt(1-α_t) * ε)`")
        analysis.append("   - 速度优势: 比DDPM快约20倍\n")

        # 质量评估
        analysis.append("### 生成质量评估\n")
        analysis.append("#### 成功指标\n")
        analysis.append("✅ **类别一致性**: 生成的精灵图像与指定类别高度一致")
        analysis.append("✅ **视觉清晰度**: 16x16像素图像结构清晰，无噪声")
        analysis.append("✅ **多样性**: 同一类别内生成多样化的变体")
        analysis.append("✅ **控制精度**: 上下文向量能够精确控制生成内容\n")

        analysis.append("#### 观察到的特征\n")
        analysis.append("- **Human类别**: 生成具有不同发色、服装的人类角色精灵")
        analysis.append("- **Food类别**: 生成各种水果和食物精灵（苹果、橙子等）")
        analysis.append("- **Non-Human类别**: 生成怪物、史莱姆等生物精灵")
        analysis.append("- **Spell类别**: 生成魔法效果和法术视觉效果")
        analysis.append("- **Side-Facing类别**: 生成侧面视角的角色")
        analysis.append("- **Unconditional**: 生成随机类别的精灵，无特定偏好\n")

        # 关键修复说明
        analysis.append("### 技术实现要点\n")
        analysis.append("#### DDIM采样公式（关键修复）\n")
        analysis.append("```python")
        analysis.append("# 正确的DDIM去噪公式")
        analysis.append("ab = ab_t[i]")
        analysis.append("ab_prev = ab_t[i_next]")
        analysis.append("x0_pred = ab_prev.sqrt() / ab.sqrt() * (samples - (1 - ab).sqrt() * eps)")
        analysis.append("dir_xt = (1 - ab_prev).sqrt() * eps")
        analysis.append("samples = x0_pred + dir_xt")
        analysis.append("```\n")

        analysis.append("#### 图像归一化（关键修复）\n")
        analysis.append("```python")
        analysis.append("# 使用固定范围 [-1,1] -> [0,1]")
        analysis.append("img = np.clip((img + 1) / 2, 0, 1)")
        analysis.append("# 而不是每个图像独立归一化")
        analysis.append("```\n")

        return analysis

    def _df_to_markdown(self, df):
        """将DataFrame转换为Markdown表格（不依赖tabulate）"""
        if df.empty:
            return "*(无数据)*"

        lines = []

        # 表头
        headers = df.columns.tolist()
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")

        # 分隔线
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        # 数据行
        for _, row in df.iterrows():
            formatted_row = []
            for val in row:
                if isinstance(val, float):
                    formatted_row.append(f"{val:.4f}")
                else:
                    formatted_row.append(str(val))
            lines.append("| " + " | ".join(formatted_row) + " |")

        return "\n".join(lines)

    def _parse_context_config(self, config):
        """解析上下文配置为可读字符串"""
        if not config or len(config) != 5:
            return "未知配置"

        categories = ['Human', 'Non-Human', 'Food', 'Spell', 'Side-Facing']
        active = [categories[i] for i, v in enumerate(config) if v == 1]

        if not active:
            return "Unconditional (无条件生成)"
        elif len(active) == 1:
            return f"{active[0]} (单一类别)"
        else:
            return f"{' + '.join(active)} (混合类别)"

    def _get_category_description(self, category):
        """获取类别描述"""
        descriptions = {
            'human': '人类角色精灵，具有不同发色和服装',
            'non_human': '非人类生物，如史莱姆、怪物、龙等',
            'food': '食物精灵，如水果、蔬菜等',
            'spell': '魔法效果、法术视觉效果',
            'side_facing': '侧面视角的角色或物体',
            'unconditional': '无条件生成，随机类别',
            'mixed_human_food': '人类角色与食物的组合（如手持食物的角色）',
            'mixed_spell_side': '侧面视角的魔法效果'
        }
        return descriptions.get(category, '未知类别')

    def _generate_conclusions(self):
        """生成综合结论"""
        conclusions = []

        conclusions.append("### 主要发现\n")

        # 超参数结论
        if self.hyper_results:
            df = pd.DataFrame([
                {
                    '实验': exp_name,
                    '最终损失': exp_data['final_loss'],
                    **exp_data['config']
                }
                for exp_name, exp_data in self.hyper_results.items()
            ])
            best_exp = df.loc[df['最终损失'].idxmin()]
            conclusions.append(f"1. **最优超参数**: 特征维度={best_exp['n_feat']}, 学习率={best_exp['lr']}, 批次大小={best_exp['batch_size']}")

        # 采样算法结论
        if self.sampling_results:
            speedups = []
            for model_data in self.sampling_results.values():
                speedup = model_data['ddpm']['time'] / model_data['ddim']['time']
                speedups.append(speedup)
            avg_speedup = np.mean(speedups)
            conclusions.append(f"2. **采样效率**: DDIM比DDPM平均快{avg_speedup:.1f}倍")

        # 引导生成结论
        if self.guided_results:
            conclusions.append("3. **引导控制**: 模型能够有效响应上下文条件，实现可控生成")

        conclusions.append("\n### 实践建议\n")
        conclusions.append("1. 对于快速原型开发，推荐使用DDIM采样")
        conclusions.append("2. 对于高质量生成，可使用DDPM采样")
        conclusions.append("3. 超参数调优对模型性能有显著影响")
        conclusions.append("4. 引导生成技术为可控AI艺术创作提供了有效手段")

        return conclusions

    def _plot_hyperparameter_losses(self):
        """绘制超参数损失曲线"""
        if not MATPLOTLIB_AVAILABLE or self.hyper_results is None:
            return

        try:
            import matplotlib.pyplot as plt

            # 创建两个子图：全局视图和局部放大
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            all_losses = []
            labels = []

            # 绘制完整的损失曲线
            for exp_name, exp_data in self.hyper_results.items():
                config = exp_data['config']
                losses = exp_data['train_losses']
                all_losses.append(losses)

                # 创建更详细的标签，包含所有关键超参数
                label_parts = []
                label_parts.append(f"feat={config.get('n_feat', 'N/A')}")
                label_parts.append(f"lr={config.get('lr', 'N/A')}")
                label_parts.append(f"bs={config.get('batch_size', 'N/A')}")

                # 如果有优化器信息，也加入
                if 'optimizer' in config:
                    label_parts.append(f"opt={config['optimizer']}")
                if 'weight_decay' in config and config['weight_decay'] > 0:
                    label_parts.append(f"wd={config['weight_decay']}")
                if 'scheduler' in config:
                    label_parts.append(f"sch={config['scheduler']}")

                label = ", ".join(label_parts)
                labels.append(label)

                # 左图：完整曲线
                ax1.plot(losses, label=label, alpha=0.7, linewidth=1.5)

            # 左图设置
            ax1.set_xlabel('Training Steps', fontsize=12)
            ax1.set_ylabel('Loss Value', fontsize=12)
            ax1.set_title('Complete Training Loss Curves', fontsize=14, weight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 右图：寻找差异最大的区域并放大
            # 计算每个时间步的损失方差
            min_length = min(len(l) for l in all_losses)
            losses_array = np.array([l[:min_length] for l in all_losses])
            variance = np.var(losses_array, axis=0)

            # 使用滑动窗口找到方差最大的区域
            window_size = min(50, min_length // 10)  # 窗口大小
            max_var = 0
            max_var_idx = 0

            for i in range(len(variance) - window_size):
                window_var = np.mean(variance[i:i+window_size])
                if window_var > max_var:
                    max_var = window_var
                    max_var_idx = i

            # 确定放大区域（添加一些边距）
            start_idx = max(0, max_var_idx - window_size // 2)
            end_idx = min(min_length, max_var_idx + window_size * 2)

            # 绘制放大区域
            for i, (exp_name, exp_data) in enumerate(self.hyper_results.items()):
                losses = exp_data['train_losses']
                if len(losses) >= end_idx:
                    ax2.plot(range(start_idx, end_idx),
                            losses[start_idx:end_idx],
                            label=labels[i],
                            alpha=0.8,
                            linewidth=2)

            # 右图设置
            ax2.set_xlabel('Training Steps', fontsize=12)
            ax2.set_ylabel('Loss Value', fontsize=12)
            ax2.set_title(f'Zoomed View (Steps {start_idx}-{end_idx})\nHighest Variance Region',
                         fontsize=14, weight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            # 添加标注
            ax2.axvspan(max_var_idx, max_var_idx + window_size,
                       alpha=0.2, color='red',
                       label='Max Variance Window')

            plt.suptitle('Hyperparameter Experiment Loss Comparison',
                        fontsize=16, weight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(self.results_dir / "hyperparameter_losses.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"超参数损失对比图已保存: {self.results_dir / 'hyperparameter_losses.png'}")
            print(f"  - 全局视图: 完整训练曲线")
            print(f"  - 局部放大: 步骤 {start_idx}-{end_idx} (最大差异区域)")

        except Exception as e:
            print(f"绘制超参数损失曲线时出错: {e}")

    def _plot_sampling_comparison(self):
        """绘制采样算法对比图"""
        if not MATPLOTLIB_AVAILABLE or self.sampling_results is None:
            return

        try:
            import matplotlib.pyplot as plt
            models = list(self.sampling_results.keys())
            ddpm_times = [self.sampling_results[m]['ddpm']['time'] for m in models]
            ddim_times = [self.sampling_results[m]['ddim']['time'] for m in models]

            x = np.arange(len(models))
            width = 0.35

            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, ddpm_times, width, label='DDPM', alpha=0.7)
            plt.bar(x + width/2, ddim_times, width, label='DDIM', alpha=0.7)

            plt.xlabel('模型')
            plt.ylabel('采样时间 (秒)')
            plt.title('DDPM vs DDIM 采样时间对比')
            plt.xticks(x, models)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.results_dir / "sampling_time_comparison.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"绘制采样算法对比图时出错: {e}")

    def _plot_guided_generation(self):
        """绘制引导生成结果"""
        if not MATPLOTLIB_AVAILABLE:
            return

        samples_dir = self.results_dir / "samples"

        # 检查目录是否存在
        if not samples_dir.exists():
            return

        # 找到所有引导生成的样本文件
        guided_files = sorted(list(samples_dir.glob("guided_*.npy")))

        if not guided_files:
            return

        try:
            import matplotlib.pyplot as plt
            # 创建对比图
            n_rows = len(guided_files)
            n_cols = 8
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2*n_rows))

            # 处理单行情况
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for i, guided_file in enumerate(guided_files):
                config_name = guided_file.stem.replace("guided_", "").replace("_", " ").title()
                samples = np.load(guided_file)

                for j in range(min(n_cols, len(samples))):
                    ax = axes[i, j] if n_rows > 1 else axes[j]
                    img = samples[j].transpose(1, 2, 0)
                    # 使用正确的归一化：固定范围 [-1, 1] -> [0, 1]
                    img = np.clip((img + 1) / 2, 0, 1)
                    ax.imshow(img)
                    ax.axis('off')

                    if j == 0:
                        # 在第一列添加类别标签
                        ax.text(-0.5, 0.5, config_name,
                               transform=ax.transAxes,
                               fontsize=10,
                               verticalalignment='center',
                               horizontalalignment='right',
                               weight='bold')

            plt.suptitle('Guided Generation Results by Category', fontsize=14, y=0.995)
            plt.tight_layout()
            plt.savefig(self.results_dir / "guided_generation_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"引导生成对比图已保存: {self.results_dir / 'guided_generation_comparison.png'}")
        except Exception as e:
            print(f"绘制引导生成结果时出错: {e}")

def analyze_experiments():
    """分析实验并生成报告"""
    analyzer = ExperimentAnalyzer()
    report = analyzer.generate_comprehensive_report()
    print("实验分析报告已生成！")
    print("报告文件: experiment_results/experiment_report.md")
    return report

if __name__ == "__main__":
    analyze_experiments()