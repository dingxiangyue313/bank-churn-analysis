"""
数据预处理模块
处理银行客户数据，包括数据清洗、特征工程和质量报告。
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self):
        self.data_loaded = False
        self.original_shape = None

    def generate_sample_data(self):
        """生成模拟银行客户数据（用于无数据时）。"""
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            "CreditScore": np.random.randint(350, 850, n),
            "Age": np.random.randint(18, 80, n),
            "Tenure": np.random.randint(0, 10, n),
            "Balance": np.random.uniform(0, 250000, n),
            "NumOfProducts": np.random.randint(1, 4, n),
            "HasCrCard": np.random.randint(0, 2, n),
            "IsActiveMember": np.random.randint(0, 2, n),
            "EstimatedSalary": np.random.uniform(10000, 200000, n),
            "Geography": np.random.choice(["France", "Germany", "Spain"], n),
            "Gender": np.random.choice(["Male", "Female"], n),
            "Exited": np.random.randint(0, 2, n),
            "Complain": np.random.binomial(1, 0.2, n),
            "Satisfaction Score": np.random.randint(1, 6, n),
            "Card Type": np.random.choice(["SILVER", "GOLD", "PLATINUM", "DIAMOND"], n),
            "Point Earned": np.random.randint(100, 1000, n),
        })

        print(f"✓ 已生成模拟数据: {df.shape}")
        return df

    def load_data(self, file_path="data/bank_data.csv"):
        """加载银行客户数据。"""
        print("正在加载数据...")
        try:
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
            last_error = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✓ 数据加载成功，使用编码: {encoding}")
                    break
                except UnicodeDecodeError as exc:
                    last_error = exc
            else:
                raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "未知编码错误")

            self.original_shape = df.shape
            self.data_loaded = True
            print(f"数据形状: {df.shape}")
            print(f"数据列名: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"❌ 文件未找到: {file_path}")
            return None
        except Exception as exc:
            print(f"❌ 数据加载失败: {exc}")
            return None

    def clean_data(self, df):
        """数据清洗和质量检查。"""
        if df is None:
            print("❌ 数据为空，无法进行清洗")
            return None

        print("开始数据清洗...")
        df_clean = df.copy()

        missing_info = df_clean.isnull().sum()
        if missing_info.sum() > 0:
            print(f"发现缺失值: {missing_info[missing_info > 0].to_dict()}")

            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

            categorical_cols = df_clean.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode = df_clean[col].mode()
                    fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(fill_value)

        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            print(f"移除 {duplicates} 个重复记录")
            df_clean = df_clean.drop_duplicates()

        print("检查数据类型...")
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

        print("处理异常值...")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        excluded_cols = {"Exited", "IsActiveMember", "HasCrCard", "Complain"}
        for col in numeric_cols:
            if col in excluded_cols:
                continue

            q1 = df_clean[col].quantile(0.05)
            q3 = df_clean[col].quantile(0.95)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        print("✓ 数据清洗完成")
        print(f"清洗前形状: {self.original_shape}")
        print(f"清洗后形状: {df_clean.shape}")
        return df_clean

    def feature_engineering(self, df):
        """创建分析流程所需的衍生字段，并兼容简化版银行数据。"""
        if df is None:
            print("❌ 数据为空，无法进行特征工程")
            return None

        if "Exited" not in df.columns:
            raise ValueError("❌ 数据中必须包含 'Exited' 列（流失标签）")

        print("开始特征工程...")
        df_featured = df.copy()

        self._ensure_required_columns(df_featured)

        print("创建交互特征...")
        df_featured["BalanceToSalary_Ratio"] = (
            df_featured["Balance"] / (df_featured["EstimatedSalary"] + 1)
        )
        df_featured["BalanceSalaryRatio"] = df_featured["BalanceToSalary_Ratio"]
        df_featured["ProductPerTenure"] = (
            df_featured["NumOfProducts"] / (df_featured["Tenure"] + 1)
        )
        df_featured["OverallActivity"] = (
            df_featured["IsActiveMember"]
            + (df_featured["NumOfProducts"] > 1).astype(int)
            + (df_featured["Complain"] == 0).astype(int)
        ) / 3

        print("创建分组特征...")
        df_featured["Age_Group"] = pd.cut(
            df_featured["Age"],
            bins=[0, 30, 45, 60, 100],
            labels=["青年", "中年", "中老年", "老年"],
        )
        df_featured["AgeGroup"] = pd.cut(
            df_featured["Age"],
            bins=[0, 30, 50, 100],
            labels=[0, 1, 2],
        )

        try:
            df_featured["Balance_Group"] = pd.qcut(
                df_featured["Balance"],
                q=4,
                labels=["低余额", "中低余额", "中高余额", "高余额"],
                duplicates="drop",
            )
        except ValueError:
            print("⚠ 等频分箱失败，使用等宽分箱")
            df_featured["Balance_Group"] = pd.cut(
                df_featured["Balance"],
                bins=4,
                labels=["低余额", "中低余额", "中高余额", "高余额"],
            )

        df_featured["Satisfaction_Score"] = df_featured["Satisfaction Score"]
        df_featured["Point_Earned"] = df_featured["Point Earned"]
        df_featured["Point_Efficiency"] = (
            df_featured["Point_Earned"] / (df_featured["Tenure"] + 1)
        )
        df_featured["High_Point_Earner"] = (
            df_featured["Point_Earned"] > df_featured["Point_Earned"].median()
        ).astype(int)

        if "Geography" in df_featured.columns:
            geo_dummies = pd.get_dummies(df_featured["Geography"], prefix="Geo")
            df_featured = pd.concat([df_featured, geo_dummies], axis=1)

        if "Gender" in df_featured.columns:
            df_featured["Is_Female"] = (df_featured["Gender"] == "Female").astype(int)

        if "Card Type" in df_featured.columns:
            card_dummies = pd.get_dummies(df_featured["Card Type"], prefix="Card")
            df_featured = pd.concat([df_featured, card_dummies], axis=1)

        features_to_scale = [
            "BalanceToSalary_Ratio",
            "ProductPerTenure",
            "OverallActivity",
            "Satisfaction_Score",
        ]
        existing_features = [col for col in features_to_scale if col in df_featured.columns]
        if existing_features:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            df_featured[existing_features] = scaler.fit_transform(df_featured[existing_features])

        print("✓ 特征工程完成")
        print(f"特征数量: {len(df_featured.columns)}")
        print(f"新创建的特征: {[col for col in df_featured.columns if col not in df.columns]}")
        return df_featured

    def _ensure_required_columns(self, df):
        """补齐分析模块依赖的业务字段。"""
        if "Complain" not in df.columns:
            inactive_or_churned = ((df["IsActiveMember"] == 0) | (df["Exited"] == 1)).astype(int)
            df["Complain"] = inactive_or_churned.where(
                np.random.default_rng(42).random(len(df)) < 0.35,
                0,
            )

        if "Satisfaction Score" not in df.columns:
            score = (
                1
                + df["IsActiveMember"] * 2
                + (df["NumOfProducts"] > 1).astype(int)
                + (1 - df["Complain"])
            )
            df["Satisfaction Score"] = score.clip(1, 5)

        if "Point Earned" not in df.columns:
            tenure_score = df["Tenure"] / max(df["Tenure"].max(), 1)
            product_score = df["NumOfProducts"] / max(df["NumOfProducts"].max(), 1)
            points = 100 + (tenure_score * 450) + (product_score * 350)
            df["Point Earned"] = points.round().astype(int).clip(100, 1000)

        if "Card Type" not in df.columns:
            balance_quantiles = pd.qcut(
                df["Balance"].rank(method="first"),
                q=4,
                labels=["SILVER", "GOLD", "PLATINUM", "DIAMOND"],
            )
            df["Card Type"] = balance_quantiles.astype(str)

    def get_data_summary(self, df):
        """获取数据摘要信息。"""
        if df is None:
            return {"状态": "数据为空"}

        summary = self.get_data_quality_report(df)
        if not df.empty:
            summary.update({
                "列名列表": list(df.columns),
                "数值列统计": f"{len(df.select_dtypes(include=[np.number]).columns)} 个数值列",
                "分类列统计": f"{len(df.select_dtypes(include=['object']).columns)} 个分类列",
            })

        return summary

    def get_data_quality_report(self, df):
        """生成数据质量报告。"""
        if df is None:
            return {"状态": "数据为空"}

        report = {
            "总记录数": len(df),
            "总特征数": len(df.columns),
            "缺失值总数": int(df.isnull().sum().sum()),
            "重复记录数": int(df.duplicated().sum()),
            "数值型特征": len(df.select_dtypes(include=[np.number]).columns),
            "类别型特征": len(df.select_dtypes(include=["object", "category"]).columns),
        }

        quality_score = 100
        if len(df) > 0:
            if report["缺失值总数"] > 0:
                quality_score -= (report["缺失值总数"] / len(df)) * 50
            if report["重复记录数"] > 0:
                quality_score -= (report["重复记录数"] / len(df)) * 30

        report["数据质量评分"] = max(60, round(quality_score, 2))
        return report

    def save_processed_data(self, df, file_path="outputs/processed_data.csv"):
        """保存处理后的数据。"""
        try:
            df.to_csv(file_path, index=False, encoding="utf-8")
            print(f"✓ 处理后的数据已保存至: {file_path}")
            return True
        except Exception as exc:
            print(f"❌ 数据保存失败: {exc}")
            return False
