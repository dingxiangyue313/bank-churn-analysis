"""
单个项目数据分析 - 输出到 singleProjectData.txt
"""

import pandas as pd
from data_preprocessing import DataPreprocessor
from micro_analysis import MicroAnalysis
from rfm_analysis import RFMAnalyzer

def run_single_project_analysis(data_path, output_file='singleProjectData.txt'):
    """
    运行单个项目数据分析并输出结果
    
    Args:
        data_path: 数据文件路径
        output_file: 输出文件名
    """
    
    print(f"开始单个项目数据分析...")
    print(f"数据文件: {data_path}")
    print(f"输出文件: {output_file}")
    
    try:
        # 加载数据
        df = pd.read_csv(data_path)
        print(f"✓ 数据加载成功，形状: {df.shape}")

        required_columns = {'Complain', 'Customer_Segment', 'Point Earned'}
        if not required_columns.issubset(df.columns):
            print("检测到原始数据，正在补充特征和客户分群...")
            preprocessor = DataPreprocessor()
            preprocessor.original_shape = df.shape
            df_clean = preprocessor.clean_data(df)
            df = preprocessor.feature_engineering(df_clean)

            if 'Customer_Segment' not in df.columns:
                analyzer = RFMAnalyzer()
                df = analyzer.create_adaptive_rfm(df)
            print(f"✓ 单项分析数据准备完成，形状: {df.shape}")
        
        # 运行微观分析
        analyzer = MicroAnalysis()
        results = analyzer.run_all_analysis(df)
        
        if results:
            # 生成详细报告
            report = analyzer.generate_micro_analysis_report()
            
            # 添加文件头信息
            full_report = f"""单个项目数据分析报告
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
数据文件: {data_path}
数据规模: {len(df):,} 条记录
整体流失率: {df['Exited'].mean():.1%}

{report}

=== 分析总结 ===
基于微观分析框架，我们实现了：
1. 高价值客户流失预警 - 识别静默风险客户
2. 防流失产品加固 - 发现产品持有缺口  
3. 地域风险洞察 - 定位高风险地区
4. 客户风险画像 - 勾勒典型流失特征
5. 服务体验分析 - 量化投诉影响
6. 活跃度监控 - 预警行为异动

这些分析为精准挽留策略提供了数据支撑。
"""
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            print(f"✓ 分析完成！结果已保存至: {output_file}")
            return True
        else:
            print("❌ 分析失败")
            return False
            
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 可以直接运行这个脚本进行独立分析
    data_file = 'data/bank_data.csv'  # 根据需要修改路径
    run_single_project_analysis(data_file)
