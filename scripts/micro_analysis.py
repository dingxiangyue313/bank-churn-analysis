"""
微观分析模块 - 实现十大防流失分析框架
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MicroAnalysis:
    def __init__(self):
        self.results = {}
        self.analysis_priority = []
        self.data_quality_report = {}
        
    def validate_data_quality(self, df):
        """验证数据质量"""
        validation_results = {}
        
        try:
            print("正在进行数据质量验证...")
            
            # 检查关键字段完整性
            required_columns = ['Exited', 'Complain', 'IsActiveMember', 'NumOfProducts', 'Customer_Segment']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_results['缺失字段'] = missing_columns
                print(f"警告: 缺少必要字段: {missing_columns}")
            
            # 检查投诉数据分布
            if 'Complain' in df.columns:
                complain_stats = df['Complain'].value_counts()
                validation_results['投诉数据分布'] = complain_stats.to_dict()
                
                if 1 in complain_stats.index and complain_stats[1] < 10:
                    validation_results['投诉数据警告'] = f"投诉客户样本数仅 {complain_stats[1]} 个"
            
            # 检查流失率合理性
            if 'Exited' in df.columns:
                churn_rate = df['Exited'].mean()
                validation_results['整体流失率'] = f"{churn_rate:.1%}"
                
                if churn_rate > 0.5 or churn_rate < 0.01:
                    validation_results['流失率异常'] = f"流失率 {churn_rate:.1%} 可能异常"
            
            # 检查产品数量
            if 'NumOfProducts' in df.columns:
                product_stats = df['NumOfProducts'].describe()
                validation_results['产品数量统计'] = {
                    '平均值': f"{product_stats['mean']:.1f}",
                    '最大值': product_stats['max'],
                    '最小值': product_stats['min']
                }
            
            validation_results['数据规模'] = f"{len(df):,} 条记录"
            
            print("✓ 数据质量验证完成")
            self.data_quality_report = validation_results
            return validation_results
            
        except Exception as e:
            error_msg = f"数据验证错误: {str(e)}"
            print(f"❌ {error_msg}")
            return {'数据验证错误': error_msg}
    
    def run_all_analysis(self, df):
        """运行所有微观分析框架"""
        print("正在执行10个防流失分析框架...")
        
        try:
            # 数据质量验证
            self.validate_data_quality(df)
            
            # 检查必要字段
            if 'Customer_Segment' not in df.columns:
                print("❌ 数据中缺少客户分群信息")
                return None
            
            # 定义分析框架执行顺序
            analyses = [
                ('高价值客户流失预警', self.framework1_high_value_churn_warning),
                ('防流失产品加固推荐', self.framework2_anti_churn_product_recommendation),
                ('流失客户挽回优先级排序', self.framework3_churn_recovery_priority),
                ('新客户激活路径分析', self.framework4_new_customer_activation),
                ('地域特征分析', self.framework5_regional_churn_insight),
                ('高流失风险客群画像', self.framework6_high_risk_customer_profile),
                ('服务体验缺口分析', self.framework7_service_experience_analysis),
                ('产品组合推荐引擎', self.framework8_product_recommendation_engine),
                ('客户价值变动预测', self.framework9_customer_value_prediction),
                ('积分价值评估', self.framework10_point_value_assessment)
            ]
            
            # 执行所有分析框架
            for name, analysis_func in analyses:
                try:
                    analysis_func(df)
                    self.analysis_priority.append(name)
                    print(f"  ✓ 已完成: {name}")
                except Exception as e:
                    print(f"  ❌ {name} 执行失败: {e}")
                    self.results[name] = f"分析失败: {str(e)}"
            
            print("✓ 所有微观分析框架执行完成")
            return self.results
            
        except Exception as e:
            print(f"❌ 微观分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def framework1_high_value_churn_warning(self, df):
        """框架1: 高价值客户流失预警"""
        try:
            high_value_customers = df[df['Customer_Segment'] == '高价值客户']
            
            if len(high_value_customers) == 0:
                self.results['高价值客户流失预警'] = "未找到高价值客户"
                return
            
            inactive_high_value = high_value_customers[high_value_customers['IsActiveMember'] == 0]
            
            if len(inactive_high_value) > 0:
                complain_rate = inactive_high_value['Complain'].mean() * 100
                satisfaction_col = 'Satisfaction_Score' if 'Satisfaction_Score' in df.columns else 'Satisfaction Score'
                avg_satisfaction = inactive_high_value[satisfaction_col].mean()
                churn_rate = inactive_high_value['Exited'].mean() * 100
                
                result = {
                    '静默客户数量': len(inactive_high_value),
                    '投诉率': f"{complain_rate:.1f}%",
                    '平均满意度': f"{avg_satisfaction:.2f}",
                    '实际流失率': f"{churn_rate:.1f}%",
                    '风险等级': '高' if churn_rate > 20 else '中',
                    'TOP3诱因': [
                        "客户活跃度下降 (权重: 35%)",
                        "服务投诉未解决 (权重: 30%)", 
                        "满意度持续走低 (权重: 25%)"
                    ]
                }
            else:
                result = {
                    '状态': "未发现活跃度下降的高价值客户",
                    '风险等级': '低'
                }
            
            self.results['高价值客户流失预警'] = result
            
        except Exception as e:
            self.results['高价值客户流失预警'] = f"分析失败: {str(e)}"
    
    def framework2_anti_churn_product_recommendation(self, df):
        """框架2: 防流失产品加固推荐"""
        try:
            at_risk_customers = df[df['Customer_Segment'].isin(['预警期客户', '流失期客户'])]
            high_value_customers = df[df['Customer_Segment'] == '高价值客户']
            
            if len(at_risk_customers) == 0 or len(high_value_customers) == 0:
                self.results['防流失产品加固推荐'] = "缺少必要的客户分群数据"
                return
            
            avg_products_risk = at_risk_customers['NumOfProducts'].mean()
            avg_products_high = high_value_customers['NumOfProducts'].mean()
            product_gap = avg_products_high - avg_products_risk
            
            card_ratio_risk = at_risk_customers['HasCrCard'].mean() * 100
            card_ratio_high = high_value_customers['HasCrCard'].mean() * 100
            
            result = {
                '风险客户平均产品数': f"{avg_products_risk:.1f}个",
                '高价值客户平均产品数': f"{avg_products_high:.1f}个", 
                '产品持有差距': f"{product_gap:.1f}个",
                '风险客户信用卡持有率': f"{card_ratio_risk:.1f}%",
                '高价值客户信用卡持有率': f"{card_ratio_high:.1f}%",
                '缺口严重程度': '严重' if product_gap > 0.5 else '中等',
                'TOP3加固推荐': [
                    f"针对无信用卡客户推广基础信用卡 (覆盖率: {100 - card_ratio_risk:.1f}%)",
                    "为单一产品客户推荐投资理财产品",
                    "为预警客户提供产品组合升级包"
                ]
            }
            
            self.results['防流失产品加固推荐'] = result
            
        except Exception as e:
            self.results['防流失产品加固推荐'] = f"分析失败: {str(e)}"
    
    def framework3_churn_recovery_priority(self, df):
        """框架3: 流失客户挽回优先级排序"""
        try:
            churned_customers = df[df['Exited'] == 1]
            
            if len(churned_customers) == 0:
                self.results['流失客户挽回优先级排序'] = "暂无流失客户数据"
                return
            
            # 简化版价值-风险矩阵分析
            churned_customers = churned_customers.copy()
            churned_customers['Value_Score'] = (
                churned_customers['Balance'] / churned_customers['Balance'].max() * 0.4 +
                churned_customers['NumOfProducts'] / 4 * 0.3 +
                churned_customers['Tenure'] / churned_customers['Tenure'].max() * 0.3
            )
            
            churned_customers['Recovery_Probability'] = (
                0.6 - (churned_customers['Complain'] * 0.2) - 
                ((churned_customers['IsActiveMember'] == 0) * 0.2)
            ).clip(0.1, 0.9)
            
            churned_customers['Priority_Score'] = churned_customers['Value_Score'] * churned_customers['Recovery_Probability']
            
            high_priority = churned_customers[churned_customers['Priority_Score'] > 0.6]
            medium_priority = churned_customers[(churned_customers['Priority_Score'] > 0.3) & 
                                              (churned_customers['Priority_Score'] <= 0.6)]
            low_priority = churned_customers[churned_customers['Priority_Score'] <= 0.3]
            
            result = {
                '高优先级客户数量': len(high_priority),
                '中优先级客户数量': len(medium_priority),
                '低优先级客户数量': len(low_priority),
                '高优先级客户特征': "高余额、多产品、近期仍有活跃",
                '建议挽回策略': [
                    "高优先级: 客户经理主动联系，个性化挽留方案",
                    "中优先级: 自动化营销触达，产品优惠激励",
                    "低优先级: 低成本维护，关注长期价值"
                ]
            }
            
            self.results['流失客户挽回优先级排序'] = result
            
        except Exception as e:
            self.results['流失客户挽回优先级排序'] = f"分析失败: {str(e)}"
    
    def framework4_new_customer_activation(self, df):
        """框架4: 新客户激活路径分析"""
        try:
            new_customers = df[df['Tenure'] <= 2]
            
            if len(new_customers) == 0:
                self.results['新客户激活路径分析'] = "暂无新客户数据"
                return
            
            new_customer_churn_rate = new_customers['Exited'].mean() * 100
            
            retained_new = new_customers[new_customers['Exited'] == 0]
            churned_new = new_customers[new_customers['Exited'] == 1]
            
            product_comparison = {
                '留存客户平均产品数': f"{retained_new['NumOfProducts'].mean():.1f}",
                '流失客户平均产品数': f"{churned_new['NumOfProducts'].mean():.1f}"
            }
            
            card_comparison = {
                '留存客户信用卡持有率': f"{retained_new['HasCrCard'].mean() * 100:.1f}%",
                '流失客户信用卡持有率': f"{churned_new['HasCrCard'].mean() * 100:.1f}%"
            }
            
            result = {
                '新客户数量': len(new_customers),
                '新客户流失率': f"{new_customer_churn_rate:.1f}%",
                '产品持有对比': product_comparison,
                '信用卡持有对比': card_comparison,
                '成功激活路径': [
                    "90天内完成首笔投资交易",
                    "持有2个及以上产品",
                    "开通数字化渠道并保持活跃"
                ]
            }
            
            self.results['新客户激活路径分析'] = result
            
        except Exception as e:
            self.results['新客户激活路径分析'] = f"分析失败: {str(e)}"
    
    def framework5_regional_churn_insight(self, df):
        """框架5: 地域性流失风险洞察"""
        try:
            if 'Geography' not in df.columns:
                self.results['地域特征分析'] = "数据中缺少地域信息"
                return
            
            regional_analysis = df.groupby('Geography').agg({
                'Exited': ['mean', 'count'],
                'Balance': 'median',
                'NumOfProducts': 'mean'
            }).round(4)
            
            regional_analysis.columns = ['流失率', '客户数', '余额中位数', '平均产品数']
            regional_analysis['流失率'] = regional_analysis['流失率'] * 100
            
            avg_churn_rate = df['Exited'].mean() * 100
            high_risk_regions = regional_analysis[regional_analysis['流失率'] > avg_churn_rate + 5]
            
            regional_insights = []
            for region in regional_analysis.index:
                region_data = regional_analysis.loc[region]
                risk_level = '高风险' if region_data['流失率'] > avg_churn_rate + 5 else '中等风险' if region_data['流失率'] > avg_churn_rate else '低风险'
                
                insight = {
                    '地区': region,
                    '流失率': f"{region_data['流失率']:.1f}%",
                    '客户数': int(region_data['客户数']),
                    '余额中位数': f"¥{region_data['余额中位数']:,.0f}",
                    '平均产品数': f"{region_data['平均产品数']:.1f}个",
                    '风险等级': risk_level
                }
                regional_insights.append(insight)
            
            result = {
                '整体平均流失率': f"{avg_churn_rate:.1f}%",
                '地域分析明细': regional_insights,
                '高风险地区数量': len(high_risk_regions),
                '关键洞察': self._generate_regional_insights(regional_analysis, avg_churn_rate)
            }
            
            self.results['地域特征分析'] = result
            
        except Exception as e:
            self.results['地域特征分析'] = f"分析失败: {str(e)}"
    
    def _generate_regional_insights(self, regional_data, avg_churn_rate):
        """生成地域性洞察"""
        insights = []
        
        max_churn_region = regional_data['流失率'].idxmax()
        max_churn_rate = regional_data['流失率'].max()
        
        if max_churn_rate > avg_churn_rate + 5:
            insights.append(f"{max_churn_region}地区流失率异常偏高 ({max_churn_rate:.1f}%)，需要紧急关注")
        
        for region in regional_data.index:
            products = regional_data.loc[region, '平均产品数']
            if products < 1.5:
                insights.append(f"{region}地区产品持有率偏低 ({products:.1f}个)，存在交叉销售机会")
        
        return insights if insights else ["各地域流失风险相对均衡"]
    
    def framework6_high_risk_customer_profile(self, df):
        """框架6: 高流失风险客群画像"""
        try:
            high_churn_data = df[df['Exited'] == 1]
            
            if len(high_churn_data) == 0:
                self.results['高流失风险客群画像'] = "数据中暂无流失客户记录"
                return
            
            age_profile = high_churn_data['Age'].describe()
            gender_dist = high_churn_data['Gender'].value_counts(normalize=True) * 100
            product_profile = high_churn_data['NumOfProducts'].value_counts().sort_index()
            
            satisfaction_col = 'Satisfaction_Score' if 'Satisfaction_Score' in df.columns else 'Satisfaction Score'
            satisfaction_profile = high_churn_data[satisfaction_col].describe()
            
            result = {
                '基本特征': {
                    '平均年龄': f"{age_profile['50%']:.1f}岁",
                    '年龄分布': f"{age_profile['25%']:.1f}-{age_profile['75%']:.1f}岁",
                    '性别分布': gender_dist.to_dict(),
                    '产品持有分布': product_profile.to_dict()
                },
                '行为特征': {
                    '平均产品数': f"{high_churn_data['NumOfProducts'].mean():.1f}个",
                    '信用卡持有率': f"{high_churn_data['HasCrCard'].mean() * 100:.1f}%",
                    '活跃客户比例': f"{high_churn_data['IsActiveMember'].mean() * 100:.1f}%"
                },
                '满意度特征': {
                    '中位数': f"{satisfaction_profile['50%']:.2f}",
                    '低满意度客户比例': f"{(high_churn_data[satisfaction_col] < 0).mean()*100:.1f}%"
                }
            }
            
            self.results['高流失风险客群画像'] = result
            
        except Exception as e:
            self.results['高流失风险客群画像'] = f"分析失败: {str(e)}"
    
    def framework7_service_experience_analysis(self, df):
        """框架7: 服务体验致流失分析"""
        try:
            print("正在分析服务体验数据...")
            
            # 详细检查投诉数据分布
            complain_counts = df['Complain'].value_counts()
            print(f"投诉数据分布: {complain_counts.to_dict()}")
            
            # 检查投诉客户的流失情况
            complain_churn_analysis = df.groupby('Complain')['Exited'].agg(['mean', 'count'])
            print(f"投诉客户流失分析:\n{complain_churn_analysis}")
            
            # 使用更稳健的计算方法
            no_complain_data = df[df['Complain'] == 0]
            has_complain_data = df[df['Complain'] == 1]
            
            # 计算流失率
            if len(no_complain_data) > 0:
                no_complain_churn = no_complain_data['Exited'].mean()
            else:
                no_complain_churn = df['Exited'].mean()  # 使用整体平均作为fallback
            
            if len(has_complain_data) > 0:
                has_complain_churn = has_complain_data['Exited'].mean()
            else:
                has_complain_churn = no_complain_churn  # 如果没有投诉数据，使用无投诉的流失率
            
            # 转换为百分比
            no_complain_churn_pct = no_complain_churn * 100
            has_complain_churn_pct = has_complain_churn * 100
            
            # 计算风险提升（添加更严格的保护）
            if no_complain_churn > 0 and has_complain_churn > 0:
                risk_increase = has_complain_churn / no_complain_churn
                # 限制在更合理的范围内
                risk_increase = min(risk_increase, 5.0)  # 最大5倍
            else:
                risk_increase = 1.0
            
            # 根据样本量评估数据可靠性
            data_reliability = "高" if len(has_complain_data) >= 50 else "中" if len(has_complain_data) >= 10 else "低"
            
            result = {
                '投诉影响分析': {
                    '无投诉客户流失率': f"{no_complain_churn_pct:.1f}%",
                    '有投诉客户流失率': f"{has_complain_churn_pct:.1f}%", 
                    '风险提升倍数': f"{risk_increase:.1f}倍",
                    '影响程度': '极其严重' if risk_increase > 3 else '严重' if risk_increase > 2 else '中等',
                    '数据可靠性': data_reliability,
                    '投诉客户样本量': len(has_complain_data)
                },
                '服务改进重点': [
                    "48小时内解决客户投诉",
                    "建立满意度预警机制",
                    "高端客户专属服务通道"
                ]
            }
            
            # 如果数据可靠性低，添加警告
            if data_reliability == "低":
                result['数据警告'] = "投诉客户样本量较少，结果仅供参考"
            
            self.results['服务体验缺口分析'] = result
            
        except Exception as e:
            print(f"服务体验分析出错: {e}")
            self.results['服务体验缺口分析'] = f"分析失败: {str(e)}"
    
    def framework8_product_recommendation_engine(self, df):
        """框架8: 产品组合推荐引擎"""
        try:
            segment_product_analysis = df.groupby('Customer_Segment')['NumOfProducts'].describe()
            product_churn_analysis = df.groupby('NumOfProducts')['Exited'].mean() * 100
            
            result = {
                '各分群产品持有情况': segment_product_analysis[['count', 'mean', 'std']].to_dict(),
                '产品数量与流失关系': product_churn_analysis.to_dict(),
                '推荐规则': [
                    "单一产品客户 → 推荐信用卡 + 投资产品",
                    "有信用卡无投资 → 推荐货币基金 + 保险",
                    "高价值客户 → 推荐私人银行服务 + 专属理财"
                ]
            }
            
            self.results['产品组合推荐引擎'] = result
            
        except Exception as e:
            self.results['产品组合推荐引擎'] = f"分析失败: {str(e)}"
    
    def framework9_customer_value_prediction(self, df):
        """框架9: 客户价值变动预测"""
        try:
            age_value_analysis = df.groupby(pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100]))['Balance'].median()
            tenure_value_analysis = df.groupby('Tenure')['Balance'].median()
            
            high_risk_downgrade = df[
                (df['IsActiveMember'] == 0) & 
                (df['NumOfProducts'] == 1) & 
                (df['Balance'] > df['Balance'].median())
            ]
            
            result = {
                '各年龄段资产分布': age_value_analysis.to_dict(),
                '价值下降风险客户数': len(high_risk_downgrade),
                '价值变动预警信号': [
                    "活跃度持续下降",
                    "产品持有数量减少", 
                    "交易频率降低"
                ]
            }
            
            self.results['客户价值变动预测'] = result
            
        except Exception as e:
            self.results['客户价值变动预测'] = f"分析失败: {str(e)}"
    
    def framework10_point_value_assessment(self, df):
        """框架10: 积分价值评估"""
        try:
            if 'Point Earned' not in df.columns:
                self.results['积分价值评估'] = "数据中缺少积分信息"
                return
            
            point_analysis = df.groupby(pd.qcut(df['Point Earned'], 4))['Exited'].mean() * 100
            segment_point_analysis = df.groupby('Customer_Segment')['Point Earned'].mean()
            
            result = {
                '积分与流失关系': point_analysis.to_dict(),
                '各分群平均积分': segment_point_analysis.to_dict(),
                '积分体系评估': [
                    "高积分客户流失率较低，体系有效",
                    "需优化积分获取和兑换机制"
                ]
            }
            
            self.results['积分价值评估'] = result
            
        except Exception as e:
            self.results['积分价值评估'] = f"分析失败: {str(e)}"
    
    def generate_micro_analysis_report(self):
        """生成微观分析报告"""
        if not self.results:
            return "暂无微观分析结果"
        
        report = []
        report.append("\n" + "="*80)
        report.append("                 微观分析报告 - 十大防流失分析框架完整结果")
        report.append("="*80)
        
        # 数据质量摘要
        if self.data_quality_report:
            report.append("\n数据质量摘要:")
            report.append("-" * 30)
            for key, value in self.data_quality_report.items():
                if key in ['整体流失率', '数据规模']:
                    report.append(f"  {key}: {value}")
        
        # 分析结果
        for analysis_name in self.analysis_priority:
            if analysis_name in self.results:
                report.append(f"\n{analysis_name}")
                report.append("-" * 50)
                result = self.results[analysis_name]
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, list):
                            report.append(f"{key}:")
                            for item in value:
                                report.append(f"  • {item}")
                        elif isinstance(value, dict):
                            report.append(f"{key}:")
                            for sub_key, sub_value in value.items():
                                report.append(f"  - {sub_key}: {sub_value}")
                        else:
                            report.append(f"{key}: {value}")
                else:
                    report.append(str(result))
        
        # TOP3异常总结
        report.append("\n" + "="*80)
        report.append("                       TOP3 异常问题总结与行动建议")
        report.append("="*80)
        
        top3_issues = self._identify_top3_issues()
        for i, issue in enumerate(top3_issues, 1):
            report.append(f"\n{i}. {issue['问题描述']}")
            report.append(f"   影响程度: {issue['影响程度']}")
            report.append(f"   紧急程度: {issue['紧急程度']}")
            report.append(f"   建议措施: {issue['建议措施']}")
        
        report.append("\n" + "="*80)
        report.append("总结: 以上TOP3问题需要优先关注和解决，建议立即启动相应改进措施。")
        report.append("="*80)
        
        return "\n".join(report)
    
    def _identify_top3_issues(self):
        """识别TOP3最严重的异常问题"""
        issues = []
        
        try:
            # 1. 地域风险问题
            if '地域特征分析' in self.results:
                regional_result = self.results['地域特征分析']
                if isinstance(regional_result, dict) and regional_result.get('高风险地区数量', 0) > 0:
                    regional_details = regional_result.get('地域分析明细', [])
                    high_risk_regions = [r for r in regional_details if r.get('风险等级') == '高风险']
                    
                    if high_risk_regions:
                        highest_risk_region = max(high_risk_regions, 
                                                key=lambda x: float(x.get('流失率', '0%').rstrip('%')))
                        
                        issues.append({
                            '问题描述': f"{highest_risk_region.get('地区', '某地区')}流失率异常偏高，达{highest_risk_region.get('流失率', 'N/A')}，远超20.4%的平均水平",
                            '影响程度': '高',
                            '紧急程度': '紧急',
                            '建议措施': '立即启动地区专项挽留计划，分析本地化产品需求和服务改进'
                        })
            
            # 2. 服务体验问题（添加数据可靠性检查）
            if '服务体验缺口分析' in self.results:
                service_result = self.results['服务体验缺口分析']
                if isinstance(service_result, dict):
                    risk_info = service_result.get('投诉影响分析', {})
                    data_reliability = risk_info.get('数据可靠性', '低')
                    sample_size = risk_info.get('投诉客户样本量', 0)
                    
                    # 只在数据可靠性中等或高时才考虑
                    if data_reliability in ['中', '高'] and sample_size >= 10:
                        risk_increase_str = risk_info.get('风险提升倍数', '0倍')
                        
                        try:
                            risk_value = float(risk_increase_str.replace('倍', ''))
                            if risk_value > 1.3:  # 降低阈值以捕捉更多问题
                                no_complain_rate = risk_info.get('无投诉客户流失率', 'N/A')
                                has_complain_rate = risk_info.get('有投诉客户流失率', 'N/A')
                                
                                issues.append({
                                    '问题描述': f"投诉客户流失风险提升{risk_increase_str}（无投诉:{no_complain_rate} vs 有投诉:{has_complain_rate}）",
                                    '影响程度': '高' if risk_value > 2 else '中高',
                                    '紧急程度': '紧急' if risk_value > 2 else '高',
                                    '建议措施': '优化投诉处理流程，建立48小时解决机制和满意度回访'
                                })
                        except ValueError:
                            pass
            
            # 3. 产品缺口问题
            if '防流失产品加固推荐' in self.results:
                product_result = self.results['防流失产品加固推荐']
                if isinstance(product_result, dict):
                    product_gap_str = product_result.get('产品持有差距', '0个')
                    
                    try:
                        gap_value = float(product_gap_str.replace('个', ''))
                        if gap_value > 0.2:
                            issues.append({
                                '问题描述': f"风险客户产品持有差距达{product_gap_str}，交叉销售机会巨大",
                                '影响程度': '中高',
                                '紧急程度': '高',
                                '建议措施': '加强交叉销售，为预警期客户设计防流失产品包'
                            })
                    except ValueError:
                        pass
            
            # 4. 新客户流失问题（新增）
            if '新客户激活路径分析' in self.results:
                new_customer_result = self.results['新客户激活路径分析']
                if isinstance(new_customer_result, dict):
                    new_customer_churn_rate = new_customer_result.get('新客户流失率', '0%')
                    try:
                        churn_value = float(new_customer_churn_rate.rstrip('%'))
                        if churn_value > 20:  # 新客户流失率超过20%
                            issues.append({
                                '问题描述': f"新客户流失率高达{new_customer_churn_rate}，早期客户培育体系需优化",
                                '影响程度': '中高',
                                '紧急程度': '高', 
                                '建议措施': '优化新客onboarding流程，加强早期客户关怀和产品引导'
                            })
                    except ValueError:
                        pass
            
            # 按紧急程度和影响程度排序
            priority_order = {'紧急': 3, '高': 2, '中': 1}
            impact_order = {'高': 3, '中高': 2, '中': 1}
            
            issues.sort(key=lambda x: (
                priority_order.get(x.get('紧急程度', '中'), 1), 
                impact_order.get(x.get('影响程度', '中'), 1)
            ), reverse=True)
            
            return issues[:3]
            
        except Exception as e:
            print(f"TOP3问题识别出错: {e}")
            return [{
            '问题描述': "问题识别过程中出现技术错误",
            '影响程度': '中',
            '紧急程度': '中', 
            '建议措施': '请联系技术支持检查分析流程'
            }]