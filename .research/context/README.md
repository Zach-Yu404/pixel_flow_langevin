# Shared Facts

机械性观察写在这里，一次采集、双方复用，避免两个 agent 重复做同样的
ls / dependency discovery / test run / log summary。

格式（`facts.md`，追加式）：

```
## {{DATE}} · {{采集者}}
- Shared Fact: `solver.py` 当前只在 stage endpoint 调用 CG。（来源：grep + 阅读 solver.py:88-120）
- Shared Fact: 测试共 21 个，`pytest -q` 3.2s 全过。（commit abc1234）
```

注意：这里只放**事实**。对事实的分析和判断必须各 agent 独立完成，不在此同步。
