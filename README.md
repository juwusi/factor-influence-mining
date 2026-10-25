# Factor Influence Mining（影响因素挖掘）

一个用于 WorkBuddy 的技能（Skill）：从结构化数据（CSV/Excel）和可量化的文本信息中，**找出对目标变量影响最显著的因素**，并输出可复用的 Python 分析脚本。

> 与相关性分析不同，本技能回答的是「哪些因素影响最大」——统计关联 / 预测贡献的排序。若要严格因果推断（判断 A 是否真的导致 B），请使用配套技能 [causal-inference](https://github.com/juwusi/causal-inference)。

## 功能特性

- 自动识别列类型（数值 / 分类 / 文本 / ID / 时间），排除 ID、时间等噪声列
- 文本列自动量化为关键词特征（TF-IDF），再纳入建模
- 单因素筛选（相关 / ANOVA F / 卡方）快速剔除无关变量
- 多因素建模：随机森林 / XGBoost 重要性 + 线性系数定「正/负影响方向」
- 自动提示多重共线性（高度相关的特征可能是同一因素的两种表达）
- 输出因素重要性排序表 + JSON 结果 + 可视化图表

## 安装

将本仓库克隆到 WorkBuddy 的用户级技能目录：

```bash
# Windows
git clone https://github.com/juwusi/factor-influence-mining.git %USERPROFILE%\.workbuddy\skills\factor-influence-mining

# macOS / Linux
git clone https://github.com/juwusi/factor-influence-mining.git ~/.workbuddy/skills/factor-influence-mining
```

也可以在 WorkBuddy 对话中直接说「安装 factor-influence-mining 技能」。

## 使用

在 WorkBuddy 对话中触发，例如：

- 「什么因素影响房价」「哪些指标驱动了留存率」
- 「帮我找出这张表里对销量影响最大的因素」

也可以直接运行核心脚本：

```bash
# 探查数据
python scripts/factor_mining.py inspect data.csv

# 分析影响因素
python scripts/factor_mining.py analyze data.csv \
  --target 房价_万 \
  --text-cols 评论 \
  --topk 10 \
  --plot
```

## 依赖

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`
- 可选：`xgboost`, `matplotlib`

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

## 方法说明

| 目标变量类型 | 单因素检验 | 首选模型 |
|---|---|---|
| 连续（价格、销量） | Pearson/Spearman 相关 | 随机森林/XGBoost 回归 |
| 分类（是否流失） | ANOVA F / 卡方 | 随机森林/XGBoost 分类 |

详细方法选型与解读见 [`references/methodology.md`](references/methodology.md)。

## 注意

输出的是**统计关联 / 预测贡献**，不自动下因果结论。严格因果推断请使用配套技能 causal-inference。

## License

[MIT](LICENSE)
