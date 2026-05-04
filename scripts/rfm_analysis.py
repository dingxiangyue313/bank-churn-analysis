"""
RFM分析模块
基于适应性RFM模型进行客户分群
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

class RFMAnalyzer:
    def __init__(self, output_dir='outputs'):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.optimal_k = 4
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_adaptive_rfm(self, df):
        """
        基于现有银行数据创建适应性RFM模型
        """
        print("创建适应性RFM模型...")
        
        # 1. 创建替代性RFM指标
        # R (Recency替代): 使用客户活跃度和关系时长
        df['R_Score'] = df['IsActiveMember'] * (1 + df['Tenure'] / df['Tenure'].max())
        
        # F (Frequency替代): 使用产品持有数量和互动指标
        df['F_Score'] = (
            df['NumOfProducts'] / df['NumOfProducts'].max() + 
            df['Point_Earned'] / df['Point_Earned'].max()
        ) / 2
        
        # M (Monetary替代): 使用余额和预估收入
        df['M_Score'] = (
            df['Balance'] / df['Balance'].max() + 
            df['EstimatedSalary'] / df['EstimatedSalary'].max()
        ) / 2
        
        # 2. 标准化RFM分数
        rfm_features = ['R_Score', 'F_Score', 'M_Score']
        df[rfm_features] = self.scaler.fit_transform(df[rfm_features])
        
        # 3. 确定最佳K值
        self._find_optimal_k(df[rfm_features])
        
        # 4. 应用K-means聚类
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=42)
        df['RFM_Cluster'] = self.kmeans_model.fit_predict(df[rfm_features])
        
        # 5. 分配业务标签
        df['Customer_Segment'] = df.apply(self._assign_customer_segment, axis=1)
        
        print("✓ RFM分析完成")
        return df
    
    def _find_optimal_k(self, X, max_k=10):
        """使用肘部法则和轮廓系数确定最佳K值"""
        print("寻找最佳聚类数量...")

        max_k = min(max_k, len(X) - 1)
        if max_k < 2:
            self.optimal_k = 1
            print("样本量不足，跳过最佳K值搜索")
            return
        
        wcss = []  # 簇内平方和
        silhouette_scores = []
        ch_scores = []  # Calinski-Harabasz分数
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            ch_scores.append(calinski_harabasz_score(X, labels))
        
        # 简单选择轮廓系数最高的K值
        best_silhouette_idx = np.argmax(silhouette_scores)
        self.optimal_k = range(2, max_k + 1)[best_silhouette_idx]
        
        print(f"最佳聚类数量: {self.optimal_k} (轮廓系数: {silhouette_scores[best_silhouette_idx]:.3f})")
        
        # 可视化评估指标
        self._plot_k_selection(range(2, max_k + 1), wcss, silhouette_scores, ch_scores)
    
    def _plot_k_selection(self, k_range, wcss, silhouette_scores, ch_scores):
        """绘制K值选择评估图"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # 肘部法则图
        ax1.plot(k_range, wcss, 'bo-')
        ax1.set_xlabel('簇数量 (K)')
        ax1.set_ylabel('簇内平方和 (WCSS)')
        ax1.set_title('肘部法则')
        ax1.axvline(self.optimal_k, color='red', linestyle='--', alpha=0.7)
        
        # 轮廓系数图
        ax2.plot(k_range, silhouette_scores, 'go-')
        ax2.set_xlabel('簇数量 (K)')
        ax2.set_ylabel('轮廓系数')
        ax2.set_title('轮廓系数评估')
        ax2.axvline(self.optimal_k, color='red', linestyle='--', alpha=0.7)
        
        # CH指数图
        ax3.plot(k_range, ch_scores, 'mo-')
        ax3.set_xlabel('簇数量 (K)')
        ax3.set_ylabel('CH指数')
        ax3.set_title('Calinski-Harabasz指数')
        ax3.axvline(self.optimal_k, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'k_selection_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _assign_customer_segment(self, row):
        """基于RFM分数分配客户分群标签"""
        r_score = row['R_Score']
        f_score = row['F_Score'] 
        m_score = row['M_Score']
        
        # 更严格的分群规则
        if m_score > 0.7 and f_score > 0.5 and r_score > 0.6:
            return '高价值客户'
        elif r_score > 0.4 and f_score > 0.2 and m_score > 0.1:
            return '发展期客户'
        elif r_score < -0.3 and f_score < -0.2:
            return '流失期客户'
        elif m_score < -0.1 or f_score < -0.1:
            return '预警期客户'
        else:
            return '稳定期客户'  # 新增一个中间群体
    
    def analyze_segment_performance(self, df):
        """分析各客户分群的业务表现"""
        print("分析客户分群表现...")
        
        segment_analysis = df.groupby('Customer_Segment').agg({
            'Exited': ['mean', 'count'],
            'Balance': 'mean',
            'EstimatedSalary': 'mean', 
            'NumOfProducts': 'mean',
            'Satisfaction_Score': 'mean',
            'Complain': 'mean',
            'Tenure': 'mean'
        }).round(4)
        
        # 重命名列
        segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns.values]
        segment_analysis = segment_analysis.rename(columns={'Exited_count': '客户数量'})
        segment_analysis['占比'] = segment_analysis['客户数量'] / len(df)
        
        print("\n客户分群分析结果:")
        print(segment_analysis)
        
        return segment_analysis
    
    def get_segment_strategies(self):
        """为每个客户分群生成策略建议"""
        strategies = {
            '高价值客户': {
                '特征': '高余额、高收入、多产品持有、高活跃度',
                '流失风险': '中等',
                '核心目标': '提升忠诚度，增加交叉销售',
                '推荐策略': [
                    '专属客户经理1对1服务',
                    '高级产品优先体验权', 
                    '个性化利率优惠方案',
                    '高端增值服务包（机场贵宾厅等）',
                    '定期财富管理咨询'
                ],
                '沟通频率': '每月至少1次主动联系',
                '预算分配': '高'
            },
            '发展期客户': {
                '特征': '活跃度高，有成长潜力，价值中等',
                '流失风险': '中低', 
                '核心目标': '促进产品使用，提升客户价值',
                '推荐策略': [
                    '产品使用引导和培训',
                    '定期关怀沟通和需求挖掘',
                    '适度的促销和激励活动',
                    '积分奖励和会员升级计划',
                    '个性化产品推荐'
                ],
                '沟通频率': '每季度2-3次联系',
                '预算分配': '中'
            },
            '预警期客户': {
                '特征': '活跃度下降，价值中等，有流失风险',
                '流失风险': '高',
                '核心目标': '防止流失，重新激活',
                '推荐策略': [
                    '流失预警外呼和调研',
                    '重新激活专项活动',
                    '产品组合优化建议',
                    '费用减免和特殊优惠',
                    '满意度提升计划'
                ],
                '沟通频率': '每月2-3次密集联系',
                '预算分配': '中高'
            },
            '流失期客户': {
                '特征': '低价值，低活跃度，高流失概率',
                '流失风险': '极高',
                '核心目标': '控制成本，选择性保留',
                '推荐策略': [
                    '低成本自动化维护',
                    '自助服务引导和优化',
                    '基础产品保留方案',
                    '自然淘汰管理',
                    '资源重新分配'
                ],
                '沟通频率': '低频率联系',
                '预算分配': '低'
            }
        }
        
        return strategies

if __name__ == "__main__":
    analyzer = RFMAnalyzer()
    
    # 测试代码
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.feature_engineering(df_clean)
    
    df_rfm = analyzer.create_adaptive_rfm(df_features)
    segment_stats = analyzer.analyze_segment_performance(df_rfm)
    strategies = analyzer.get_segment_strategies()
    
    print("\n策略建议生成完成!")
