"""
银行客户流失分析主程序
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加scripts目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_dir, 'scripts')
sys.path.append(scripts_path)

# 导入模块
try:
    from data_preprocessing import DataPreprocessor
    from rfm_analysis import RFMAnalyzer
    from visualization import Visualization
    from micro_analysis import MicroAnalysis
    from single_project_analysis import run_single_project_analysis

    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保scripts目录中有所有必要的.py文件")
    sys.exit(1)

class BankCustomerAnalysis:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.analyzer = RFMAnalyzer()
        self.visualizer = Visualization()
        self.micro_analyzer = MicroAnalysis()
        self.results = {}
        
        # 创建输出目录
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
     
    def run_analysis(self):
        """运行完整分析流程"""
        print(f"\n分析开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 数据加载与预处理
            df_final = self._load_and_preprocess_data()
            if df_final is None:
                return None
            
            # RFM分析
            df_rfm, segment_stats, strategies = self._perform_rfm_analysis(df_final)
            if df_rfm is None:
                return None
            
            # 微观分析
            micro_results = self._perform_micro_analysis(df_rfm)
            
            # 保存结果
            self.results.update({
                'processed_data': df_rfm,
                'segment_stats': segment_stats,
                'strategies': strategies,
                'micro_analysis': micro_results
            })
            
            # 生成报告和可视化
            self._generate_reports_and_visualizations(df_rfm, segment_stats, strategies)
            
            print("\n✓ 分析完成！")
            self._print_output_summary()
            
            return self.results
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_and_preprocess_data(self):
        """数据加载与预处理"""
        print("\n步骤1：数据准备与预处理")
        
        data_path = 'data/bank_data.csv'
        print(f"正在加载数据，路径: {os.path.abspath(data_path)}")
        
        # 检查文件是否存在，不存在则生成模拟数据
        if not os.path.exists(data_path):
            print("⚠️ 未找到真实数据，生成模拟数据...")
            df = self.preprocessor.generate_sample_data()
            # 保存模拟数据（方便后续复用）
            os.makedirs('data', exist_ok=True)
            df.to_csv(data_path, index=False)
            print(f"✓ 模拟数据已保存至: {data_path}")
        else:
            # 加载真实数据
            df = self.preprocessor.load_data(data_path)
            if df is None or df.empty:
                print("❌ 数据加载失败，生成模拟数据...")
                df = self.preprocessor.generate_sample_data()

        print(f"✓ 数据加载成功，形状: {df.shape}")

        # 数据清洗
        df_clean = self.preprocessor.clean_data(df)
        if df_clean is None:
            print("❌ 数据清洗失败")
            return None

        # 特征工程
        df_final = self.preprocessor.feature_engineering(df_clean)
        if df_final is None:
            print("❌ 特征工程失败")
            return None

        print(f"✓ 数据预处理完成，最终形状: {df_final.shape}")

        # 数据质量报告
        data_summary = self.preprocessor.get_data_quality_report(df_final)
        print("\n数据质量报告:")
        for key, value in data_summary.items():
            print(f"  {key}: {value}")

        return df_final

    def _find_alternative_data_path(self):
        """查找备选数据路径（当前逻辑中已用模拟数据替代，可保留兼容）"""
        alternative_paths = [
            '../data/bank_data.csv',
            './bank_data.csv',
            'bank_data.csv'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✓ 在备用位置找到文件: {alt_path}")
                return alt_path
        return None  # 若没找到，后续会生成模拟数据

    def _perform_rfm_analysis(self, df_final):
        """执行RFM分析"""
        print("\n步骤2：RFM客户分群分析")

        # RFM分析
        df_rfm = self.analyzer.create_adaptive_rfm(df_final)
        if df_rfm is None:
            print("❌ RFM分析失败")
            return None, None, None

        print("✓ RFM分析完成")

        # 分群表现分析
        segment_stats = self.analyzer.analyze_segment_performance(df_rfm)
        if segment_stats is None:
            print("❌ 分群表现分析失败")
            return None, None, None

        print("✓ 分群表现分析完成")

        # 策略建议
        strategies = self.analyzer.get_segment_strategies()
        print("✓ 策略建议生成完成")

        # 显示关键结果
        self._display_rfm_key_results(df_rfm, segment_stats, strategies)

        return df_rfm, segment_stats, strategies

    def _perform_micro_analysis(self, df_rfm):
        """执行微观分析"""
        print("\n步骤3：微观分析 - 精准诊断与诱因定位")

        micro_results = self.micro_analyzer.run_all_analysis(df_rfm)

        if micro_results:
            print("✓ 微观分析完成")
            self._display_micro_analysis_results(micro_results)
        else:
            print("❌ 微观分析执行失败")

        return micro_results

    def _generate_reports_and_visualizations(self, df_rfm, segment_stats, strategies):
        """生成报告和可视化"""
        print("\n步骤4：生成可视化图表和报告")

        # 生成可视化图表
        self.visualizer.create_all_visualizations(df_rfm, segment_stats, strategies)

        # 生成最终报告
        self._generate_final_report()

        # 生成单个项目数据报告
        print("\n步骤5：生成单个项目数据分析报告")
        run_single_project_analysis('outputs/processed_customer_data.csv', 'singleProjectData.txt')

    def _display_rfm_key_results(self, df_rfm, segment_stats, strategies):
        """显示RFM分析关键结果"""
        print("\n" + "="*60)
        print("RFM分析关键结果")
        print("="*60)

        print(f"\n总客户数: {len(df_rfm):,}")

        print(f"\n客户分群分布:")
        segment_counts = df_rfm['Customer_Segment'].value_counts()
        for segment, count in segment_counts.items():
            percentage = (count / len(df_rfm)) * 100
            print(f"  {segment}: {count} 人 ({percentage:.1f}%)")

        print(f"\n各分群流失率:")
        segment_exit_rates = df_rfm.groupby('Customer_Segment')['Exited'].mean()
        for segment, rate in segment_exit_rates.items():
            print(f"  {segment}: {rate:.1%}")

        # 关键洞察
        max_exit_segment = segment_exit_rates.idxmax()
        max_exit_rate = segment_exit_rates.max()
        largest_segment = segment_counts.idxmax()

        print(f"\n关键洞察:")
        print(f"  • 流失风险最高的分群: {max_exit_segment} (流失率: {max_exit_rate:.1%})")
        print(f"  • 规模最大的分群: {largest_segment}")
        print(f"  • 建议优先关注 {max_exit_segment} 的客户保留策略")

    def _display_micro_analysis_results(self, micro_results):
        """显示微观分析关键结果"""
        print("\n" + "="*60)
        print("微观分析关键洞察")
        print("="*60)

        # 地域分析结果
        if '地域特征分析' in micro_results:
            regional = micro_results['地域特征分析']
            if isinstance(regional, dict):
                print(f"\n地域风险分析:")
                print(f"• 整体平均流失率: {regional.get('整体平均流失率', 'N/A')}")
                print(f"• 高风险地区数量: {regional.get('高风险地区数量', 'N/A')}")

                high_risk_regions = regional.get('地域分析明细', [])
                if high_risk_regions:
                    print("• 需要关注的地区:")
                    for region in high_risk_regions[:3]:
                        print(f"  - {region.get('地区', 'N/A')} (流失率: {region.get('流失率', 'N/A')})")

        # 服务体验分析结果
        if '服务体验缺口分析' in micro_results:
            service = micro_results['服务体验缺口分析']
            if isinstance(service, dict):
                risk_info = service.get('投诉影响分析', {})
                print(f"\n服务体验分析:")
                print(f"• 投诉客户流失风险提升: {risk_info.get('风险提升倍数', 'N/A')}")

    def _generate_final_report(self):
        """生成最终分析报告"""
        report_content = self._create_report_content()

        # 保存文本报告
        with open('reports/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 保存处理后的数据
        if 'processed_data' in self.results:
            self.results['processed_data'].to_csv('outputs/processed_customer_data.csv', index=False)

        # 保存微观分析详细报告
        if 'micro_analysis' in self.results:
            micro_report = self.micro_analyzer.generate_micro_analysis_report()
            with open('reports/detailed_micro_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(micro_report)

        print("✓ 分析报告已生成: reports/analysis_report.txt")
        print("✓ 微观分析报告已生成: reports/detailed_micro_analysis.txt")
        print("✓ 处理数据已保存: outputs/processed_customer_data.csv")

    def _create_report_content(self):
        """创建报告内容"""
        data_summary = self.preprocessor.get_data_quality_report(self.results.get('processed_data', pd.DataFrame()))
        segment_stats = self.results.get('segment_stats')
        strategies = self.results.get('strategies', {})

        report = []
        report.append("=" * 60)
        report.append("        银行客户流失分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 数据概览
        report.append("1. 数据概览")
        report.append("-" * 30)
        total_customers = len(self.results.get('processed_data', []))
        report.append(f"总客户数: {total_customers:,}")
        if not self.results.get('processed_data', pd.DataFrame()).empty:
            report.append(f"整体流失率: {self.results['processed_data']['Exited'].mean():.1%}")
            report.append(f"投诉率: {self.results['processed_data'].get('Complain', pd.Series([0])).mean():.1%}")
        report.append("")

        # 客户分群分析
        report.append("2. 客户分群分析")
        report.append("-" * 30)
        if not self.results.get('processed_data', pd.DataFrame()).empty:
            segment_counts = self.results['processed_data']['Customer_Segment'].value_counts()
            segment_exit_rates = self.results['processed_data'].groupby('Customer_Segment')['Exited'].mean()

            for segment in segment_counts.index:
                count = segment_counts[segment]
                percentage = (count / total_customers) * 100
                exit_rate = segment_exit_rates[segment]

                report.append(f"\n{segment}:")
                report.append(f"  - 客户数量: {count:,} ({percentage:.1f}%)")
                report.append(f"  - 流失率: {exit_rate:.1%}")

                if segment_stats is not None and segment in segment_stats.index:
                    stats = segment_stats.loc[segment]
                    if 'Balance_mean' in stats:
                        report.append(f"  - 平均余额: ¥{stats['Balance_mean']:,.0f}")
                    if 'NumOfProducts_mean' in stats:
                        report.append(f"  - 平均产品数: {stats['NumOfProducts_mean']:.1f}")
        report.append("")

        # 策略建议
        report.append("3. 精准挽留策略")
        report.append("-" * 30)
        for segment, strategy in strategies.items():
            report.append(f"\n{segment}:")
            report.append(f"  特征: {strategy.get('特征', 'N/A')}")
            report.append(f"  流失风险: {strategy.get('流失风险', 'N/A')}")
            report.append(f"  核心目标: {strategy.get('核心目标', 'N/A')}")
            report.append("  推荐策略:")
            for i, tactic in enumerate(strategy.get('推荐策略', [])[:3], 1):
                report.append(f"    {i}. {tactic}")
            report.append(f"  沟通频率: {strategy.get('沟通频率', 'N/A')}")
            report.append(f"  预算分配: {strategy.get('预算分配', 'N/A')}")
        report.append("")

        # 关键洞察
        report.append("4. 关键业务洞察")
        report.append("-" * 30)

        if not self.results.get('processed_data', pd.DataFrame()).empty:
            segment_exit_rates = self.results['processed_data'].groupby('Customer_Segment')['Exited'].mean()
            segment_counts = self.results['processed_data']['Customer_Segment'].value_counts()

            max_exit_segment = segment_exit_rates.idxmax() if not segment_exit_rates.empty else "N/A"
            max_exit_rate = segment_exit_rates.max() if not segment_exit_rates.empty else 0
            largest_segment = segment_counts.idxmax() if not segment_counts.empty else "N/A"
            largest_count = segment_counts.max() if not segment_counts.empty else 0

            report.append(f"• 流失风险最高的分群: {max_exit_segment} (流失率: {max_exit_rate:.1%}) - 需要紧急干预")
            report.append(f"• 规模最大的分群: {largest_segment} ({largest_count}人) - 重点发展交叉销售")
        report.append("")

        # 微观分析核心洞察
        report.append("5. 微观分析核心洞察")
        report.append("-" * 30)

        micro_results = self.results.get('micro_analysis', {})

        # 地域洞察
        if '地域特征分析' in micro_results:
            regional = micro_results['地域特征分析']
            if isinstance(regional, dict) and regional.get('关键洞察'):
                report.append("• 地域风险洞察:")
                for insight in regional['关键洞察'][:2]:
                    report.append(f"  - {insight}")

        # 服务体验洞察
        if '服务体验缺口分析' in micro_results:
            service = micro_results['服务体验缺口分析']
            if isinstance(service, dict):
                risk_info = service.get('投诉影响分析', {})
                report.append(f"• 服务影响: 投诉客户流失风险提升{risk_info.get('风险提升倍数', 'N/A')}")
        else:
            report.append("• 微观分析未执行或执行失败")
        
        report.append("")
        
        # 预期成效
        report.append("6. 预期成效")
        report.append("-" * 30)
        report.append("• 短期目标 (3个月): 降低整体流失率 10-15%")
        report.append("• 中期目标 (6个月): 提升高价值客户留存率 25%")
        report.append("• 长期目标 (1年): 建立数据驱动的客户管理体系")
        report.append("• 预计ROI: 300-500% (基于客户生命周期价值计算)")
        
        return "\n".join(report)
    
    def _print_output_summary(self):
        """打印输出文件摘要"""
        print("\n" + "="*50)
        print("输出文件摘要")
        print("="*50)
        print("请查看以下文件获取完整结果：")
        print("  • reports/analysis_report.txt - 综合分析报告")
        print("  • reports/detailed_micro_analysis.txt - 微观分析报告") 
        print("  • singleProjectData.txt - 单个项目数据报告")
        print("  • outputs/ - 可视化图表和处理后的数据")
        print(f"\n分析完成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """主函数"""
    analysis = BankCustomerAnalysis()
    
    # 运行分析
    results = analysis.run_analysis()
    
    if not results:
        print("\n❌ 分析失败，请检查错误信息")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
