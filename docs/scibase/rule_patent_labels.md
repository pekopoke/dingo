# RulePatentFieldValidation Label 说明

## 实现位置

[`rule_patent.py`](../../dingo/model/rule/scibase/rule_patent.py)

## 标签数量统计

| 字段 | 标签数量 |
|---|---:|
| `claims` | 2 |
| `ipc_unified` | 19 |
| `document_number_unified` | 4 |
| `prior_art_references` | 5 |
| **合计** | **30** |

下表中的 Label 名是字段标签后缀。规则最终输出时会添加字段名前缀，
例如 `ipc_unified` 字段的 `missing.value` 最终输出为
`ipc_unified.missing.value`。

## claims

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `claims` 字段。 |
| `too_long` | 内容过长 | 将字段值转换为字符串后，长度超过 1,000,000 个字符时标记。当前实现不校验字段类型或 claims 内部结构。 |

## ipc_unified

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `ipc_unified` 字段。 |
| `null` | 值为 null | `ipc_unified` 的值为 `null`。 |
| `invalid_json` | JSON 格式错误 | `ipc_unified` 是字符串，但无法解析为 JSON。 |
| `not_array` | 不是数组 | `ipc_unified` 解析后不是数组。 |
| `item_not_string` | 数组项不是字符串 | `ipc_unified` 数组中至少一个元素不是字符串；异常元素不参与缺失项或多余项计数。 |
| `invalid_ipc_format` | 统一 IPC 格式不符合要求 | `ipc_unified` 中至少一个字符串不符合目标格式：四位小类号 `<A-H><01-99><A-Z>`，或 `<四位小类号> <1-999>/<2-6 位数字>`。分隔符必须是一个半角空格；主组不得补零；全零小组只能写成 `00`；`F03G 0/00` 应归一化为 `F03G`。 |
| `source_invalid.null` | 原始 IPC 为 null | 原始 `ipc` 字段存在，但值为 `null`。若原始 `ipc` 字段不存在，则仅检查 `ipc_unified` 格式，不产生此标签。 |
| `source_invalid.invalid_json` | 原始 IPC JSON 格式错误 | 原始 `ipc` 是字符串，但无法解析为 JSON。 |
| `source_invalid.not_array` | 原始 IPC 不是数组 | 原始 `ipc` 解析后不是数组。 |
| `source_invalid.item_not_string` | 原始 IPC 数组项不是字符串 | 原始 `ipc` 数组中至少一个元素不是字符串。已成功转换的其他元素仍可继续检查缺失项，但不再判定多余项。 |
| `conversion_failed.ambiguous` | 转换结果有歧义 | 原始 IPC 是无斜杠数字粘连形式，存在多个合理的主组/小组拆分结果，无法唯一转换。例如 `H01R 1200`。 |
| `conversion_failed.missing_context` | 缺少转换上下文 | 原始 IPC 是省略共享小类的数字片段，但前面没有可供继承的小类。例如数组第一项直接出现 `216/14`。 |
| `conversion_failed.unsupported_format` | 暂未支持该转换格式 | 原始 IPC 元素是字符串，但其写法不在当前转换规则覆盖范围内。该标签只表示转换器暂不支持，不表示原始 IPC 格式非法。例如 `G05F001455`。 |
| `missing.all` | 转换结果全部缺失 | 原始 `ipc` 至少能确定一个预期转换值，但 `ipc_unified` 中没有任何格式有效的 IPC 值。此标签与 `missing.value`、`missing.count` 互斥。 |
| `missing.value` | 转换值缺失 | `ipc_unified` 中存在其他有效 IPC 值，但某个由原始 `ipc` 确定的预期值一次也没有出现。 |
| `missing.count` | 转换值数量不足 | 某个预期值已在 `ipc_unified` 中出现，但实际出现次数少于原始 `ipc` 转换结果中的次数。 |
| `extra.empty_source` | 空来源存在多余值 | 原始 `ipc` 是空数组，但 `ipc_unified` 中存在格式有效的 IPC 值。 |
| `extra.value` | 存在非预期值 | `ipc_unified` 中某个格式有效的值不在原始 `ipc` 的确定转换结果中。只有全部来源项都已成功转换或按明确规则移除时才判定。 |
| `extra.count` | 转换值数量过多 | 某个值属于预期转换结果，但它在 `ipc_unified` 中的出现次数多于原始 `ipc` 转换结果中的次数。 |

## document_number_unified

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `document_number_unified` 字段。 |
| `wrong_type` | 字段类型错误 | 值不是字符串时标记。 |
| `invalid_format` | 格式不符合要求 | 值包含 ASCII 字母和数字以外的字符时标记。 |
| `incorrect_conversion` | 文献号转换错误 | 当原始 `document_number` 是当前规则支持的日本专利文献号形式时，`document_number_unified` 与确定性转换结果不一致。 |

## prior_art_references

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段或必需属性缺失 | 输入对象中不存在 `prior_art_references` 字段，或数组项缺少 `sequence`、`srep_phase`、`document_number`、`document_number_unified` 中的至少一个属性。 |
| `invalid_json` | JSON 格式错误 | 字段值是字符串，但无法解析为 JSON。 |
| `wrong_type` | 字段或数组项类型错误 | 字段解析后不是数组、数组项不是对象，或数组项中的 `document_number_unified` 不是字符串。 |
| `invalid_format` | 统一文献号格式不符合要求 | 数组项中的 `document_number_unified` 包含 ASCII 字母和数字以外的字符。 |
| `incorrect_conversion` | 文献号转换错误 | 数组项中的原始 `document_number` 可按当前日本专利规则确定转换结果，但 `document_number_unified` 与该结果不一致。 |
