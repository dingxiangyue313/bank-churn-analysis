"""
可视化模块
生成银行客户数据分析的各种图表和报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
from matplotlib import font_manager

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
preferred_fonts = [
    'Arial Unicode MS',
    'STHeiti',
    'Songti SC',
    'Hiragino Sans GB',
    'SimHei',
    'Microsoft YaHei',
    'DejaVu Sans',
]
plt.rcParams['font.sans-serif'] = [font for font in preferred_fonts if font in available_fonts]
plt.rcParams['axes.unicode_minus'] = False

class Visualization:
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def clean_output_directory(self):
        """清理输出目录中的旧文件"""
        print("清理输出目录中的旧文件...")
        
        # 删除所有图片文件
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.svg']
        for pattern in image_patterns:
            for file_path in glob.glob(os.path.join(self.output_dir, pattern)):
                try:
                    os.remove(file_path)
                    print(f"  已删除: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  无法删除 {os.path.basename(file_path)}: {e}")
        
        # 删除数据文件
        data_files = ['processed_customer_data.xlsx', 'analysis_report.txt']
        for file_name in data_files:
            file_path = os.path.join(self.output_dir, file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"  已删除: {file_name}")
                except Exception as e:
                    print(f"  无法删除 {file_name}: {e}")
        
        print("✓ 输出目录清理完成")
    
    def create_all_visualizations(self, df, segment_stats, strategies, clean_output=True):
        """创建所有可视化图表"""
        print("\n开始生成可视化图表...")
        
        # 清理旧文件
        if clean_output:
            self.clean_output_directory()
        
        try:
            # 1. RFM分析图
            print("生成RFM分析图...")
            self.create_rfm_analysis_plot(df)
            
            # 2. 相关性热力图
            print("生成相关性热力图...")
            self.create_correlation_heatmap(df)
            
            # 3. 客户分群策略卡
            print("生成客户分群策略卡...")
            self.create_segment_strategy_cards(strategies)
            
            # 4. 雷达图
            print("生成分群雷达图...")
            self.create_radar_chart(segment_stats)
            
            # 5. 总览仪表板
            print("生成总览仪表板...")
            self.create_overview_dashboard(df, segment_stats)
            
            # 6. 保存处理后的数据
            print("保存处理后的数据...")
            self.save_processed_data(df)
            
            # 7. 生成文本报告
            print("生成分析报告...")
            self.generate_analysis_report(df, segment_stats, strategies)
            
            print("✓ 所有可视化图表生成完成")
            
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    def create_rfm_analysis_plot(self, df):
        """创建RFM分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('银行客户RFM分群分析', fontsize=16, fontweight='bold')
        
        # 1. 客户分群分布
        segment_counts = df['Customer_Segment'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0, 0].set_title('客户分群分布')
        
        # 2. 各分群流失率
        segment_exit = df.groupby('Customer_Segment')['Exited'].mean().sort_values(ascending=False)
        bars = axes[0, 1].bar(segment_exit.index, segment_exit.values, color=colors[:len(segment_exit)])
        axes[0, 1].set_title('各分群流失率')
        axes[0, 1].set_ylabel('流失率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, segment_exit.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.1%}', ha='center', va='bottom')
        
        # 3. RFM分数分布
        rfm_features = ['R_Score', 'F_Score', 'M_Score']
        rfm_data = df[rfm_features].melt(var_name='RFM维度', value_name='分数')
        sns.boxplot(data=rfm_data, x='RFM维度', y='分数', ax=axes[1, 0])
        axes[1, 0].set_title('RFM分数分布')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 分群特征对比
        segment_features = df.groupby('Customer_Segment')[['Balance', 'EstimatedSalary', 'NumOfProducts']].mean()
        segment_features.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('各分群关键特征对比')
        axes[1, 1].set_ylabel('平均值')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rfm_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_heatmap(self, df):
        """创建相关性热力图"""
        # 选择数值型列进行相关性分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # 排除ID类和分类编码列
        exclude_cols = ['CustomerId', 'RowNumber', 'Exited', 'IsActiveMember', 'HasCrCard', 'Complain']
        corr_cols = [col for col in numeric_cols if col not in exclude_cols and not col.startswith(('Geo_', 'Card_'))]
        
        # 只取前15个最重要的特征，避免热力图过于拥挤
        if len(corr_cols) > 15:
            # 计算与流失的相关性，选择最重要的
            if 'Exited' in df.columns:
                exit_corr = df[corr_cols + ['Exited']].corr()['Exited'].abs().sort_values(ascending=False)
                corr_cols = exit_corr.index[1:16].tolist()  # 排除Exited自身
        
        corr_matrix = df[corr_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角掩码
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('特征相关性热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_segment_strategy_cards(self, strategies):
        """创建客户分群策略卡"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.ravel()
        fig.suptitle('客户分群策略建议卡', fontsize=20, fontweight='bold', y=0.95)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (segment, strategy) in enumerate(strategies.items()):
            if idx >= 4:  # 只显示前4个分群
                break
                
            ax = axes[idx]
            # 设置背景颜色
            ax.set_facecolor(colors[idx] + '20')  # 添加透明度
            
            # 标题
            ax.text(0.5, 0.95, segment, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[idx], alpha=0.7))
            
            # 特征描述
            ax.text(0.05, 0.85, f"特征: {strategy['特征']}", transform=ax.transAxes, 
                   fontsize=10, va='top', style='italic')
            
            # 流失风险
            risk_color = {'低': 'green', '中低': 'orange', '中等': 'orange', '高': 'red', '极高': 'darkred'}
            ax.text(0.05, 0.75, f"流失风险: {strategy['流失风险']}", transform=ax.transAxes, 
                   fontsize=11, fontweight='bold', color=risk_color.get(strategy['流失风险'], 'black'))
            
            # 核心目标
            ax.text(0.05, 0.65, f"核心目标: {strategy['核心目标']}", transform=ax.transAxes, 
                   fontsize=11, fontweight='bold')
            
            # 推荐策略
            y_pos = 0.55
            ax.text(0.05, y_pos, "推荐策略:", transform=ax.transAxes, 
                   fontsize=11, fontweight='bold')
            y_pos -= 0.05
            
            for i, tactic in enumerate(strategy['推荐策略'][:4]):  # 只显示前4个策略
                ax.text(0.08, y_pos, f"• {tactic}", transform=ax.transAxes, 
                       fontsize=9, va='top')
                y_pos -= 0.06
            
            # 沟通频率和预算
            ax.text(0.05, 0.15, f"沟通频率: {strategy['沟通频率']}", transform=ax.transAxes, 
                   fontsize=10)
            ax.text(0.05, 0.08, f"预算分配: {strategy['预算分配']}", transform=ax.transAxes, 
                   fontsize=10, fontweight='bold')
            
            # 设置坐标轴范围并隐藏边框
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # 如果分群少于4个，隐藏多余的子图
        for idx in range(len(strategies), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'strategy_cards.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_radar_chart(self, segment_stats):
        """创建分群雷达图"""
        try:
            # 准备雷达图数据
            segments = segment_stats.index.tolist()
            
            # 选择要展示的指标
            metrics = ['Balance_mean', 'EstimatedSalary_mean', 'NumOfProducts_mean', 
                      'Satisfaction_Score_mean', 'Tenure_mean']
            
            # 过滤存在的指标
            available_metrics = [m for m in metrics if m in segment_stats.columns]
            
            if len(available_metrics) < 3:
                print("⚠ 可用指标不足，跳过雷达图生成")
                return
            
            # 标准化数据 (0-1范围)
            radar_data = segment_stats[available_metrics].copy()
            for col in available_metrics:
                min_val = radar_data[col].min()
                max_val = radar_data[col].max()
                if max_val > min_val:
                    radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
                else:
                    radar_data[col] = 0.5  # 如果所有值相同，设为中间值
            
            # 设置雷达图
            angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            
            for idx, segment in enumerate(segments):
                if idx >= len(colors):
                    break
                    
                values = radar_data.loc[segment].values.tolist()
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, 'o-', linewidth=2, label=segment, color=colors[idx])
                ax.fill(angles, values, alpha=0.1, color=colors[idx])
            
            # 设置角度标签
            metric_labels = [m.replace('_mean', '').replace('_', ' ') for m in available_metrics]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            
            # 设置y轴标签位置和格式
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            
            ax.set_title('客户分群特征雷达图\n(标准化值)', size=16, fontweight='bold', pad=20)
            ax.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠ 雷达图生成失败: {e}")
    
    def create_overview_dashboard(self, df, segment_stats):
        """创建总览仪表板"""
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('银行客户数据分析总览仪表板', fontsize=20, fontweight='bold')
        
        # 创建复杂的网格布局
        gs = fig.add_gridspec(3, 4)
        
        # 1. 整体流失率 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        exit_rate = df['Exited'].mean()
        colors_pie = ['#4ECDC4', '#FF6B6B'] if exit_rate < 0.5 else ['#FF6B6B', '#4ECDC4']
        ax1.pie([1-exit_rate, exit_rate], labels=['留存', '流失'], autopct='%1.1f%%', 
                colors=colors_pie, startangle=90)
        ax1.set_title(f'整体客户流失率\n({exit_rate:.1%})')
        
        # 2. 关键指标 (右上)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        metrics_text = [
            f'总客户数: {len(df):,}',
            f'平均余额: ¥{df["Balance"].mean():,.0f}',
            f'平均产品数: {df["NumOfProducts"].mean():.1f}',
            f'平均关系时长: {df["Tenure"].mean():.1f}年',
            f'活跃客户: {df["IsActiveMember"].mean():.1%}',
            f'投诉率: {df["Complain"].mean():.1%}'
        ]
        
        for i, text in enumerate(metrics_text):
            ax2.text(0.1, 0.9 - i*0.15, text, transform=ax2.transAxes, 
                    fontsize=12, fontweight='bold', va='top')
        
        # 3. 分群规模条形图 (中左)
        ax3 = fig.add_subplot(gs[1, 0])
        segment_counts = df['Customer_Segment'].value_counts()
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax3.bar(segment_counts.index, segment_counts.values, color=colors_bar[:len(segment_counts)])
        ax3.set_title('各分群客户数量')
        ax3.set_ylabel('客户数量')
        ax3.tick_params(axis='x', rotation=45)
        
        # 在条形上添加数值
        for bar, count in zip(bars, segment_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count}', ha='center', va='bottom')
        
        # 4. 分群流失率 (中右)
        ax4 = fig.add_subplot(gs[1, 1])
        segment_exit = df.groupby('Customer_Segment')['Exited'].mean().sort_values(ascending=False)
        bars = ax4.bar(segment_exit.index, segment_exit.values, color=colors_bar[:len(segment_exit)])
        ax4.set_title('各分群流失率')
        ax4.set_ylabel('流失率')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, rate in zip(bars, segment_exit.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 5. 余额分布 (左下)
        ax5 = fig.add_subplot(gs[2, 0])
        df['Balance'].hist(bins=30, ax=ax5, color='#4ECDC4', alpha=0.7)
        ax5.set_title('客户余额分布')
        ax5.set_xlabel('余额')
        ax5.set_ylabel('客户数量')
        
        # 6. 产品数量分布 (右下)
        ax6 = fig.add_subplot(gs[2, 1])
        product_counts = df['NumOfProducts'].value_counts().sort_index()
        ax6.pie(product_counts.values, labels=product_counts.index, autopct='%1.1f%%', 
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax6.set_title('产品持有数量分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overview_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_processed_data(self, df):
        """保存处理后的数据"""
        try:
            # 保存为Excel，包含多个sheet
            with pd.ExcelWriter(os.path.join(self.output_dir, 'processed_customer_data.xlsx')) as writer:
                # 原始数据（主要特征）
                main_cols = [col for col in df.columns if not col.startswith(('Geo_', 'Card_'))]
                df[main_cols].to_excel(writer, sheet_name='客户数据', index=False)
                
                # 客户分群统计
                segment_summary = df.groupby('Customer_Segment').agg({
                    'Exited': ['count', 'mean'],
                    'Balance': 'mean',
                    'EstimatedSalary': 'mean',
                    'NumOfProducts': 'mean',
                    'Tenure': 'mean'
                }).round(3)
                segment_summary.to_excel(writer, sheet_name='分群统计')
                
                # RFM分数
                rfm_cols = ['CustomerId', 'R_Score', 'F_Score', 'M_Score', 'RFM_Cluster', 'Customer_Segment']
                if all(col in df.columns for col in rfm_cols):
                    df[rfm_cols].to_excel(writer, sheet_name='RFM分数', index=False)
            
            print("✓ 处理后的数据已保存")
            
        except Exception as e:
            print(f"❌ 数据保存失败: {e}")
    
    def generate_analysis_report(self, df, segment_stats, strategies):
        """生成文本分析报告"""
        try:
            report_path = os.path.join(self.output_dir, 'analysis_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("银行客户数据分析报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 1. 基本统计
                f.write("1. 数据概览\n")
                f.write("-" * 30 + "\n")
                f.write(f"总客户数: {len(df):,}\n")
                f.write(f"总特征数: {len(df.columns)}\n")
                f.write(f"整体流失率: {df['Exited'].mean():.2%}\n")
                f.write(f"平均余额: ¥{df['Balance'].mean():,.0f}\n")
                f.write(f"平均产品持有数: {df['NumOfProducts'].mean():.1f}\n\n")
                
                # 2. 客户分群分析
                f.write("2. 客户分群分析\n")
                f.write("-" * 30 + "\n")
                segment_counts = df['Customer_Segment'].value_counts()
                for segment, count in segment_counts.items():
                    percentage = (count / len(df)) * 100
                    exit_rate = df[df['Customer_Segment'] == segment]['Exited'].mean()
                    f.write(f"{segment}: {count}人 ({percentage:.1f}%), 流失率: {exit_rate:.2%}\n")
                f.write("\n")
                
                # 3. 分群特征对比
                f.write("3. 分群特征对比\n")
                f.write("-" * 30 + "\n")
                if not segment_stats.empty:
                    for col in segment_stats.columns:
                        if col.endswith('_mean') and not col.startswith('Exited'):
                            f.write(f"\n{col.replace('_mean', '')}:\n")
                            for segment in segment_stats.index:
                                value = segment_stats.loc[segment, col]
                                f.write(f"  {segment}: {value:,.2f}\n")
                f.write("\n")
                
                # 4. 策略建议摘要
                f.write("4. 策略建议摘要\n")
                f.write("-" * 30 + "\n")
                for segment, strategy in strategies.items():
                    f.write(f"\n{segment}:\n")
                    f.write(f"  特征: {strategy['特征']}\n")
                    f.write(f"  流失风险: {strategy['流失风险']}\n")
                    f.write(f"  核心目标: {strategy['核心目标']}\n")
                    f.write(f"  关键策略: {', '.join(strategy['推荐策略'][:3])}\n")
                    f.write(f"  预算分配: {strategy['预算分配']}\n")
                
                # 5. 关键洞察
                f.write("\n5. 关键洞察与建议\n")
                f.write("-" * 30 + "\n")
                
                # 找出流失率最高的分群
                max_exit_segment = df.groupby('Customer_Segment')['Exited'].mean().idxmax()
                max_exit_rate = df.groupby('Customer_Segment')['Exited'].mean().max()
                
                # 找出规模最大的分群
                largest_segment = df['Customer_Segment'].value_counts().idxmax()
                largest_count = df['Customer_Segment'].value_counts().max()
                
                f.write(f"• 流失风险最高的分群: {max_exit_segment} (流失率: {max_exit_rate:.2%})\n")
                f.write(f"• 规模最大的分群: {largest_segment} ({largest_count}人)\n")
                f.write(f"• 建议优先关注: {max_exit_segment} 分群的客户保留\n")
                f.write(f"• 重点发展: {largest_segment} 分群的交叉销售机会\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("报告生成完成\n")
            
            print("✓ 分析报告已生成")
            
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'CustomerId': range(1, 101),
        'Balance': np.random.normal(50000, 20000, 100),
        'EstimatedSalary': np.random.normal(60000, 25000, 100),
        'NumOfProducts': np.random.randint(1, 5, 100),
        'Tenure': np.random.randint(1, 10, 100),
        'Exited': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'IsActiveMember': np.random.choice([0, 1], 100, p=[0.3, 0.7]),
        'R_Score': np.random.normal(0, 1, 100),
        'F_Score': np.random.normal(0, 1, 100),
        'M_Score': np.random.normal(0, 1, 100),
        'Customer_Segment': np.random.choice(['高价值客户', '发展期客户', '预警期客户', '流失期客户'], 100)
    })
    
    # 创建测试统计
    test_stats = test_data.groupby('Customer_Segment').agg({
        'Balance': 'mean',
        'EstimatedSalary': 'mean',
        'NumOfProducts': 'mean',
        'Tenure': 'mean'
    })
    
    # 测试策略
    test_strategies = {
        '高价值客户': {
            '特征': '高余额、高收入、多产品',
            '流失风险': '中等',
            '核心目标': '提升忠诚度',
            '推荐策略': ['专属服务', '优先体验'],
            '沟通频率': '每月1次',
            '预算分配': '高'
        },
        '发展期客户': {
            '特征': '中等价值、有潜力',
            '流失风险': '中低',
            '核心目标': '促进使用',
            '推荐策略': ['产品引导', '定期关怀'],
            '沟通频率': '每季度2-3次',
            '预算分配': '中'
        }
    }
    
    # 测试可视化
    viz = Visualization()
    viz.create_all_visualizations(test_data, test_stats, test_strategies)
    print("测试完成!")
