# C2Rust Migration Project

## 项目概述
将 src_c/ 下的 C 代码逐模块迁移为 src_rust/ 下的惯用 Rust 代码。

## 当前阶段
第一阶段：使用 Haiku 4.5 作为翻译模型，调优 Skills 和 Prompt，建立质量基准线。

## 工作流程
1. 检查 migration_status.json 确认当前进度
2. 若无迁移计划，先运行 /c2rust-analyze
3. 按 migration_plan.json 中的 wave 顺序逐模块处理
4. 每个模块：/c2rust-translate → /c2rust-verify
5. 每完成一个模块，立即更新 migration_status.json

## 模型使用规则
- 翻译任务：必须使用 translator subagent（Haiku 4.5）
- 修复任务：使用 fixer subagent（Sonnet）
- 代码搜索/依赖分析：使用 Explore subagent（内置）
- 不要在主会话中直接翻译代码，始终委派给对应的 subagent

## 类型映射规则
- int → i32, unsigned int → u32, long → i64, size_t → usize
- char* (只读参数) → &str
- char* (拥有所有权) → String
- char[] (固定缓冲区) → [u8; N] 或 Vec<u8>
- void* → 根据语义选择泛型 T、Box<dyn Any>、或 *mut c_void（标注 unsafe）
- struct 指针 → &T（借用）/ Box<T>（独占所有权）/ Rc<T>（共享）
- 函数指针 → Fn trait（优先）/ fn pointer（性能敏感场景）
- NULL 检查 → Option<T>
- 错误码返回 → Result<T, E>
- goto 错误清理 → ? 运算符 + Drop trait

## Unsafe 策略
第一阶段允许 unsafe 块。但每个 unsafe 块必须附带注释说明为什么必要。
目标：unsafe 块数量作为质量指标之一追踪，后续阶段逐步消除。

## 异常处理规则
- 单模块修复连续失败 3 次 → 标记 "needs_human_review"，继续下一模块
- 发现循环修复（修 A 引入 B，修 B 回到 A）→ 标记 "needs_retranslation"
- 模块间接口不匹配 → 暂停，切换 Opus 重新评估依赖关系
- 遇到不可翻译的模式（内联汇编、平台特定 API）→ 标记 "manual_only"

## 翻译经验
参见 translation_notes.md。每发现可复用的翻译模式时追加记录。
当同一模式出现 3 次以上，应考虑将其固化为类型映射规则。

## 质量指标
跟踪以下指标，用于评估 Skill 调优效果：
- first_pass_rate: Haiku 翻译后直接编译通过的比例
- fix_rate: 经 fixer 修复后编译通过的比例
- avg_fix_attempts: 平均修复次数
- unsafe_count: 每模块平均 unsafe 块数量
- human_review_rate: 需要人工介入的比例
