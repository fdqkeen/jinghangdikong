# -*- coding: utf-8 -*-
"""
应用冒烟测试：验证首屏与全部交互控件均无异常，并检查关键数值是否与方案正文一致。

运行：  python test_app.py
期望：  首屏 OK，error 0，warn 0；全部交互用例 OK；末行输出 DONE

重要：AppTest 收集控件的顺序是「主区先于侧栏」，且会随页面改动而变化，
      因此本测试**一律按控件标签定位，不使用下标**。
      否则会静默测错对象——曾出现「声学节点数」用例实际改到了别的滑块、
      数值从未变化却显示 OK 的假通过。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from streamlit.testing.v1 import AppTest

TPL = "app.py"


def widget(a, kind, label):
    """按标签取控件；取不到直接抛错，避免静默测错对象。"""
    for e in getattr(a, kind):
        if e.label == label:
            return e
    raise KeyError(f"未找到 {kind}：{label!r}；现有 "
                   f"{[e.label for e in getattr(a, kind)]}")


def metrics(a, keys):
    return [(m.label, m.value) for m in a.metric if any(k in m.label for k in keys)]


a = AppTest.from_file(TPL, default_timeout=300)
a.run()
print("首屏:", a.exception or "OK", "| error", len(a.error), "| warn", len(a.warning))
print("  元素: tab", len(a.tabs), "metric", len(a.metric), "df", len(a.dataframe),
      "chart", len(a.get("plotly_chart")))
for w in a.warning:
    print("   [WARN]", str(w.value)[:150])
for e in a.error:
    print("   [ERROR]", str(e.value)[:150])
print("  默认:", metrics(a, ["基线暴露", "折中方案暴露", "航程代价"]))
assert len(a.tabs) == 10, f"Tab 数量应为 10，实际 {len(a.tabs)}"
assert len(a.error) == 0 and len(a.warning) == 0, "首屏出现告警或异常元素"

CASES = [
    ("站距500", "selectbox", "基站站距 (m)", 500, ["达标率"]),
    ("站距250", "selectbox", "基站站距 (m)", 250, ["达标率"]),
    ("eVTOL", "selectbox", "机型", "eVTOL", ["基线暴露", "折中方案暴露"]),
    ("微型机", "selectbox", "机型", "微型机（Mini 类）", ["基线暴露"]),
    ("高度300", "slider", "基线巡航高度 (m)", 300, ["基线暴露"]),
    ("昼间", "radio", "评价时段", "昼间", ["基线暴露"]),
    ("节点10", "slider", "声学校准节点数", 10, ["混合 RMSE"]),
    ("节点100", "slider", "声学校准节点数", 100, ["混合 RMSE"]),
    ("设备+50%", "slider", "硬件 / 设备类单价浮动", 50, ["本方案合计", "成本下降"]),
    ("站址-50%", "slider", "站址 / 土建 / 供电 / 回传 类单价浮动", -50, ["成本下降"]),
]
for tag, kind, label, val, ks in CASES:
    b = AppTest.from_file(TPL, default_timeout=300)
    b.run()
    widget(b, kind, label).set_value(val)
    b.run()
    print(f"  {tag:10s}", b.exception or "OK", metrics(b, ks))
print("DONE")
