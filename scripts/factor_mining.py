#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining.py — 影响因素挖掘脚本
在结构化表格数据中，找出对目标变量影响最显著的特征，并输出重要性排序。

用法：
    python factor_mining.py inspect <datafile>
    python factor_mining.py analyze <datafile> --target <列名> [选项]

选项：
    --target      目标变量列名（必填，analyze 模式）
    --text-cols   需要量化的文本列，逗号分隔（可选）
    --topk        输出重要性前 K 的因素（默认 10）
    --method      建模方法：auto | rf | xgb | linear（默认 auto）
    --plot        输出可视化图片（可选，需 matplotlib）
    --out         结果输出目录（默认 ./factor_mining_output）

依赖：pandas, numpy, scikit-learn （可选：xgboost, matplotlib）
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """按扩展名加载 CSV / Excel 数据。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="utf-8", errors="ignore")
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", encoding="utf-8", errors="ignore")
    raise ValueError(f"不支持的文件类型: {ext}")


def classify_columns(df: pd.DataFrame):
    """把列分为数值 / 分类 / 文本 / ID / 时间 五类。"""
    n_rows = len(df)
    num_cols, cat_cols, text_cols, id_cols, time_cols = [], [], [], [], []
    for c in df.columns:
        s = df[c]
        nunique = s.nunique(dropna=True)
        dtype = str(s.dtype)
        if "int" in dtype or "float" in dtype:
            if nunique <= 1:
                continue  # 常量列，稍后删除
            # 数值列也可能是 ID/序号：唯一值占比极高（>90%）且列名含 id/序号/编号 等 → 视为 ID
            if nunique / max(n_rows, 1) > 0.9 and any(k in c.lower() for k in ("id", "编号", "序号", "index", "代码", "工号")):
                id_cols.append(c)
                continue
            num_cols.append(c)
        elif "datetime" in dtype:
            time_cols.append(c)
        else:
            # 判断是否像 ID / 时间
            if nunique == len(s) and s.astype(str).str.len().max() > 8:
                id_cols.append(c)
            elif s.astype(str).str.match(r"^\d{4}[-/]\d{1,2}").all() and nunique > 1:
                time_cols.append(c)
            else:
                # 文本 vs 分类：平均长度较长视为文本
                avg_len = s.astype(str).str.len().mean()
                if avg_len > 20 or nunique > 100:
                    text_cols.append(c)
                else:
                    cat_cols.append(c)
    return num_cols, cat_cols, text_cols, id_cols, time_cols


def inspect(args):
    df = load_data(args.datafile)
    print("=" * 60)
    print(f"数据文件: {args.datafile}")
    print(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print("=" * 60)
    num_cols, cat_cols, text_cols, id_cols, time_cols = classify_columns(df)
    print(f"\n数值列 ({len(num_cols)}): {num_cols}")
    print(f"分类列 ({len(cat_cols)}): {cat_cols}")
    print(f"文本列 ({len(text_cols)}): {text_cols}")
    print(f"ID 列 (建议排除) ({len(id_cols)}): {id_cols}")
    print(f"时间列 (建议排除) ({len(time_cols)}): {time_cols}")
    print("\n缺失率与唯一值数:")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "缺失率%": (df.isna().mean() * 100).round(2),
        "唯一值": df.nunique(),
    })
    print(info.to_string())
    print("\n提示: 使用 analyze --target <列名> 进行影响因素分析")


def _quantify_text(df: pd.DataFrame, text_cols, top_n=20):
    """把文本列量化为关键词频次特征。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    new_parts = []
    for c in text_cols:
        series = df[c].fillna("").astype(str)
        vec = TfidfVectorizer(
            max_features=top_n, token_pattern=r"(?u)\b\w[\w\-]{1,}\b"
        )
        m = vec.fit_transform(series).toarray()
        words = vec.get_feature_names_out()
        part = pd.DataFrame(m, columns=[f"{c}__词频_{w}" for w in words], index=df.index)
        new_parts.append(part)
        print(f"  文本列「{c}」已量化为 {len(words)} 个关键词特征")
    return pd.concat(new_parts, axis=1) if new_parts else None


def _preprocess(df, target, text_cols):
    """特征预处理：填充缺失、编码分类、量化文本。返回特征矩阵 + 特征名列表。"""
    num_cols, cat_cols, text_cols_auto, id_cols, time_cols = classify_columns(df)
    # 用户显式指定的文本列优先
    text_cols = text_cols or text_cols_auto
    feature_cols = [c for c in df.columns if c != target and c not in id_cols and c not in time_cols]
    feature_cols = [c for c in feature_cols if c not in text_cols]

    X_parts = []
    # 数值列
    num_feats = [c for c in feature_cols if c in num_cols]
    if num_feats:
        X_num = df[num_feats].apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median())
        X_parts.append(X_num)
    # 分类列
    cat_feats = [c for c in feature_cols if c in cat_cols]
    if cat_feats:
        X_cat = df[cat_feats].astype(str).fillna("缺失")
        X_cat = pd.get_dummies(X_cat, drop_first=True)
        X_parts.append(X_cat)
    # 文本列量化
    if text_cols:
        X_txt = _quantify_text(df, text_cols)
        if X_txt is not None:
            X_parts.append(X_txt)

    X = pd.concat(X_parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    # 删除常量列
    X = X.loc[:, X.nunique() > 1]
    return X, num_feats, cat_feats, text_cols


def _rank_linear(X, y, is_classification):
    """线性模型标准化系数，可解读正负方向。"""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    if is_classification:
        m = LogisticRegression(max_iter=1000).fit(Xs, y)
        coef = m.coef_[0]
    else:
        m = LinearRegression().fit(Xs, y)
        coef = m.coef_
    return pd.Series(coef, index=X.columns).abs().sort_values(ascending=False), coef


def _rank_tree(X, y, is_classification, method="rf"):
    """树模型特征重要性。"""
    if method == "xgb":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            print("  [警告] 未安装 xgboost，回退到随机森林。")
            method = "rf"
    if method == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        model = RandomForestClassifier(n_estimators=200, random_state=42) if is_classification else RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        from xgboost import XGBClassifier, XGBRegressor

        model = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="logloss") if is_classification else XGBRegressor(n_estimators=200, random_state=42)

    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp


def analyze(args):
    df = load_data(args.datafile)
    target = args.target
    if target not in df.columns:
        sys.exit(f"错误: 目标列「{target}」不存在。可用列: {list(df.columns)}")

    y = df[target]
    # 判断 Y 类型
    if str(y.dtype) in ("object", "category", "bool") or y.nunique() <= 20:
        # 分类
        from sklearn.preprocessing import LabelEncoder
        y = y.fillna(y.mode()[0] if not y.mode().empty else "缺失")
        le = LabelEncoder().fit(y.astype(str))
        y_enc = le.transform(y.astype(str))
        is_classification = True
        y_type = f"分类（{y.nunique()} 类）"
    else:
        y_enc = y.fillna(y.median()).astype(float)
        is_classification = False
        y_type = "连续（回归）"

    print("=" * 60)
    print(f"目标变量: {target}  →  {y_type}")
    print("=" * 60)

    X, num_feats, cat_feats, text_cols = _preprocess(df, target, args.text_cols)
    print(f"\n特征矩阵: {X.shape[0]} 样本 × {X.shape[1]} 特征")

    if X.shape[1] == 0:
        sys.exit("错误: 没有可用的特征列。")

    # 单因素筛选
    print("\n[1/3] 单因素相关性筛选…")
    if is_classification:
        from sklearn.feature_selection import f_classif
        scores, pvals = f_classif(X.fillna(0), y_enc)
        univariate = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        kept = X.columns[pvals < 0.05].tolist()
        print(f"  ANOVA F 检验: {len(kept)}/{X.shape[1]} 个特征显著 (p<0.05)")
    else:
        corr = X.corrwith(pd.Series(y_enc, index=X.index)).abs().sort_values(ascending=False)
        univariate = corr
        kept = corr[corr > 0.05].index.tolist()
        print(f"  相关系数: {len(kept)}/{X.shape[1]} 个特征 |corr|>0.05")

    # 多因素建模
    print("\n[2/3] 多因素建模（重要性排序）…")
    method = args.method if args.method != "auto" else ("rf")
    importance = _rank_tree(X, y_enc, is_classification, method=method)
    _, coef = _rank_linear(X, y_enc, is_classification)

    # 方向
    direction = pd.Series(np.sign(coef), index=X.columns)
    direction = direction.map({1: "正", -1: "负", 0: "—"})

    topk = min(args.topk, X.shape[1])
    print(f"\n[3/3] 重要性 Top {topk}:")
    result = []
    print(f"  {'排名':<4}{'因素':<30}{'重要性':<10}{'方向'}")
    print("  " + "-" * 52)
    for i, (feat, val) in enumerate(importance.head(topk).items(), 1):
        print(f"  {i:<4}{feat:<30}{val:<10.4f}{direction.get(feat, '—')}")
        result.append({
            "rank": i, "factor": feat, "importance": round(float(val), 4),
            "direction": direction.get(feat, "—"),
            "univariate": round(float(univariate.get(feat, np.nan)), 4) if not pd.isna(univariate.get(feat, np.nan)) else None,
        })

    # 共线性提示
    print("\n共线性提示:")
    num_feat_cols = X.columns[:min(len(X.columns), 20)]
    corr_matrix = X[num_feat_cols].corr().abs()
    high_corr = []
    for i in range(len(num_feat_cols)):
        for j in range(i + 1, len(num_feat_cols)):
            if corr_matrix.iloc[i, j] > 0.85:
                high_corr.append((num_feat_cols[i], num_feat_cols[j], round(corr_matrix.iloc[i, j], 3)))
    if high_corr:
        for a, b, v in high_corr[:5]:
            print(f"  ⚠ {a} ↔ {b} 高度相关 (r={v})，可能是同一因素的两种表达")
    else:
        print("  未发现高度共线特征")

    # 输出
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "factor_importance.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"target": target, "y_type": y_type, "factors": result}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_json}")

    if args.plot:
        _plot(importance.head(topk), args.out)


def _plot(importance, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, max(4, len(importance) * 0.4)))
        importance.sort_values().plot.barh(ax=ax, color="#3B82F6")
        ax.set_xlabel("重要性")
        ax.set_title("因素重要性排序")
        plt.tight_layout()
        path = os.path.join(out_dir, "factor_importance.png")
        plt.savefig(path, dpi=150)
        print(f"图表已保存: {path}")
    except ImportError:
        print("  [提示] 未安装 matplotlib，跳过图表输出。")


def main():
    p = argparse.ArgumentParser(description="影响因素挖掘")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="探查数据结构")
    pi.add_argument("datafile")
    pi.set_defaults(func=inspect)

    pa = sub.add_parser("analyze", help="分析影响因素")
    pa.add_argument("datafile")
    pa.add_argument("--target", required=True)
    pa.add_argument("--text-cols", default=None, help="文本列，逗号分隔")
    pa.add_argument("--topk", type=int, default=10)
    pa.add_argument("--method", default="auto", choices=["auto", "rf", "xgb", "linear"])
    pa.add_argument("--plot", action="store_true")
    pa.add_argument("--out", default="./factor_mining_output")
    pa.set_defaults(func=analyze)

    args = p.parse_args()
    if args.cmd == "analyze" and args.text_cols:
        args.text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    args.func(args)


if __name__ == "__main__":
    main()
