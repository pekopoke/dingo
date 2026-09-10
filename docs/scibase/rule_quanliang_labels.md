# RuleQuanliangFieldValidation Label 说明

## 实现位置

[`rule_quanliang.py`](../../dingo/model/rule/scibase/rule_quanliang.py)

## 标签数量统计

| 字段 | 标签数量 |
|---|---:|
| `metadata_type` | 5 |
| `doi` | 6 |
| `isbns` | 4 |
| `isbn13` | 4 |
| `title` | 21 |
| `abstract` | 21 |
| `language` | 4 |
| `author` | 10 |
| `contributors` | 4 |
| `locations` | 6 |
| `access_is_oa` | 4 |
| `access_oa_status` | 4 |
| `access_oa_url` | 4 |
| `access_license` | 4 |
| `publication_published_date` | 5 |
| `publication_published_year` | 3 |
| `publication_venue_issn` | 4 |
| `publication_venue_biblio_volume` | 4 |
| `publication_venue_biblio_issue` | 4 |
| `publication_venue_biblio_pages` | 6 |
| `publication_pages` | 3 |
| `publication_venue_name_unified` | 5 |
| `grade_class` | 4 |
| `grade` | 5 |
| `references` | 23 |
| `related_works` | 23 |
| `citations` | 23 |
| `supplementary_material` | 4 |
| `cited_by_api_url` | 4 |
| `access_xinghe_repository_sha256` | 4 |
| `access_xinghe_repository_origin_path` | 3 |
| `access_xinghe_repository_model_name` | 4 |
| `access_xinghe_repository_model_version` | 5 |
| **合计** | **237** |

## metadata_type

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `metadata_type` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `empty` | 值为空 | 若值为空，则标记。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 2 个限定值之一：`paper`、`ebook`。 |

## doi

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `doi` 字段。 |
| `empty` | 值为空 | 若论文 DOI 为空，则标记。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `not_lowercase` | 不是小写字母 | 若 DOI 不是小写，则标记。 |
| `format_invalid` | 格式不符合要求 | DOI 去除首尾空白后不符合 `10.<4～9 位数字>/<非空后缀>`；后缀不能含空白或竖线，DOI URL 也不符合。 |
| `error_prefix` | 前缀不符合要求 | 提取 DOI 的 `/` 前缀；命中 3 个测试前缀 `10.0000`、`10.0001`、`10.5555` 时标记。 |

## isbns

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `isbns` 字段。 |
| `empty` | 值为空 | 若电子书 ISBN 列表为空，则标记。 |
| `wrong_type` | 字段类型错误 | 值不是字符串列表时标记。 |
| `invalid_format` | 格式不符合要求 | 列表中至少一个值不是有效 ISBN-10 或 ISBN-13：ISBN-10 须为 9 位数字加数字/`X` 校验位；ISBN-13 须为 13 位数字、以 `978`/`979` 开头，并通过校验位计算。 |

## isbn13

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `isbn13` 字段。 |
| `empty` | 值为空 | 若电子书 ISBN13 为空，则标记。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `invalid_format` | 格式不符合要求 | 值不是有效 ISBN-13：须为 13 位数字、以 `978` 或 `979` 开头，并通过校验位计算。 |

## title

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `title` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `html_tag.layout` | 包含 HTML 排版标签 | 包含 HTML 排版标签，如 `<i>`、`<b>`、`<p>`、`<br>`、`<sup>`、`<sub>`、`<em>`、`<strong>`、`<span>`、`<div>`、`<u>`、`<scp>`、`<tt>`，包括闭合标签和带属性的标签。 |
| `html_tag.math` | 包含 MathML 标签 | 包含 MathML 标签，如 `<math>`、`<mrow>`、`<mi>`、`<mn>`、`<mo>`、`<msub>`、`<msup>`、`<mfrac>`、`<msqrt>` 等，也匹配 `mml:` 前缀。 |
| `html_tag.xml_comment` | 包含 XML 注释 | 包含 XML/HTML 注释片段，如 `<!-- comment -->`。 |
| `html_tag.cdata` | 包含 CDATA 内容 | 包含 CDATA 区段，如 `<![CDATA[...]]>`。 |
| `html_entity.named` | 包含命名 HTML 实体 | 包含 `&名称;` 形式的命名 HTML 实体，如 `&nbsp;`、`&amp;`。 |
| `html_entity.decimal` | 包含十进制 HTML 实体 | 包含 `&#数字;` 形式的十进制 HTML 实体，如 `&#160;`。 |
| `html_entity.hex` | 包含十六进制 HTML 实体 | 包含 `&#x十六进制;` 形式的 HTML 实体，如 `&#xA0;`。 |
| `special_char.invisible` | 包含不可见字符 | 包含 U+2000～U+200F、U+202F、U+205F、U+3000、U+FEFF、U+00A0 或 U+2060～U+206F 中的不可见字符或特殊空白。 |
| `special_char.replacement` | 包含 Unicode 替换字符 | 包含 Unicode 替换字符 `�`（U+FFFD）。 |
| `special_char.control` | 包含控制字符 | 包含 U+0000～U+0008、U+000B、U+000C、U+000E～U+001F 或 U+007F；TAB、换行和回车不在此标签范围内。 |
| `special_char.markup` | 包含方括号排版标记 | 包含 `[!i]`、`[!/i]`、`[!sub]`、`[!/sup]`、`[!]`、`[○!R]`、`[○!上]`、`[○!下]` 等标记。 |
| `empty` | 值为空 | 若去除首尾空格后内容为空，则标记。 |
| `too_short` | 内容过短 | 若去除首尾空格后长度小于 5 个字符，则标记。 |
| `too_long` | 内容过长 | 若去除首尾空格后长度大于 1000 个字符，则标记。 |
| `likely_placeholder` | 疑似占位内容 | 标题去除首尾空白并忽略大小写后，等于预设占位值，如 `untitled`、`[no title]`、`unknown`、`n/a`、`null`、`tbd`、`-` 或 `.`。 |
| `encoding_error` | 编码错误 | 包含常见乱码模式，如 `�`、`锟斤拷`、`烫烫烫`、`屯屯屯`、以 `Ã`/`Â` 开头的双字节乱码、`â€™` 或 `ï»¿`。 |
| `likely_conference` | 疑似会议名称 | 标题以可选年份加 `IEEE` 开头，并包含 `proceedings`、`conference`、`symposium`、`workshop` 或 `congress`。 |
| `likely_identifier` | 疑似标识符或链接 | 整个标题仅由数字、DOI、DOI URL、HTTP(S)/`www` URL，或 `s3://`、`s3a://` 路径构成。 |

## abstract

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `abstract` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `html_tag.layout` | 包含 HTML 排版标签 | 包含 HTML 排版标签，如 `<i>`、`<b>`、`<p>`、`<br>`、`<sup>`、`<sub>`、`<em>`、`<strong>`、`<span>`、`<div>`、`<u>`、`<scp>`、`<tt>`，包括闭合标签和带属性的标签。 |
| `html_tag.math` | 包含 MathML 标签 | 包含 MathML 标签，如 `<math>`、`<mrow>`、`<mi>`、`<mn>`、`<mo>`、`<msub>`、`<msup>`、`<mfrac>`、`<msqrt>` 等，也匹配 `mml:` 前缀。 |
| `html_tag.xml_comment` | 包含 XML 注释 | 包含 XML/HTML 注释片段，如 `<!-- comment -->`。 |
| `html_tag.cdata` | 包含 CDATA 内容 | 包含 CDATA 区段，如 `<![CDATA[...]]>`。 |
| `html_entity.named` | 包含命名 HTML 实体 | 包含 `&名称;` 形式的命名 HTML 实体，如 `&nbsp;`、`&amp;`。 |
| `html_entity.decimal` | 包含十进制 HTML 实体 | 包含 `&#数字;` 形式的十进制 HTML 实体，如 `&#160;`。 |
| `html_entity.hex` | 包含十六进制 HTML 实体 | 包含 `&#x十六进制;` 形式的 HTML 实体，如 `&#xA0;`。 |
| `special_char.invisible` | 包含不可见字符 | 包含 U+2000～U+200F、U+202F、U+205F、U+3000、U+FEFF、U+00A0 或 U+2060～U+206F 中的不可见字符或特殊空白。 |
| `special_char.replacement` | 包含 Unicode 替换字符 | 包含 Unicode 替换字符 `�`（U+FFFD）。 |
| `special_char.control` | 包含控制字符 | 包含 U+0000～U+0008、U+000B、U+000C、U+000E～U+001F 或 U+007F；TAB、换行和回车不在此标签范围内。 |
| `special_char.markup` | 包含方括号排版标记 | 包含 `[!i]`、`[!/i]`、`[!sub]`、`[!/sup]`、`[!]`、`[○!R]`、`[○!上]`、`[○!下]` 等标记。 |
| `empty` | 值为空 | 若去除首尾空格后内容为空，则标记。 |
| `too_short` | 内容过短 | 若去除首尾空格后长度小于 20 个字符，则标记。 |
| `too_long` | 内容过长 | 若去除首尾空格后长度大于 6000 个字符，则标记。 |
| `likely_placeholder` | 疑似占位内容 | 摘要去除首尾空白并忽略大小写后，等于 `n/a`、`none`、`null`、`unknown`、`-` 等，或匹配 `no abstract available`、`abstract not provided`、`unavailable` 等提示语。 |
| `encoding_error` | 编码错误 | 包含常见乱码模式，如 `�`、`锟斤拷`、`烫烫烫`、`屯屯屯`、以 `Ã`/`Â` 开头的双字节乱码、`â€™` 或 `ï»¿`。 |
| `same_title` | 内容与标题重复 | 摘要与字符串类型的非空标题去除首尾空白并忽略大小写后完全相同。 |
| `likely_identifier` | 疑似标识符或链接 | 整个摘要仅由数字、DOI、DOI URL、HTTP(S)/`www` URL，或 `s3://`、`s3a://` 路径构成。 |

## language

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `language` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 ISO 639 映射表中的 8038 个语言代码之一，例如 `zh`、`en`、`fr`、`de`。 |

## author

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `author` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查值是否为作者对象列表，并逐项检查对象属性类型。 |
| `invalid_keys` | key 不符合要求 | 若作者对象字段不符合要求，则标记。 |
| `empty` | 值为空 | 若作者列表为空，则标记。 |
| `empty_name` | 名称为空 | 若至少一个作者姓名去除首尾空格后为空，则标记。 |
| `duplicated_name` | 名称重复 | 非空作者名经去除首尾空白、连续空白折叠为一个空格并执行 Unicode 不区分大小写归一化后出现重复。 |
| `multiple_names` | 单项疑似包含多个作者名 | 单个 `author[].name` 命中多人名信号，如末尾 `et al.`/“等”、连接词 `and`/“和”/“与”/“及”/顿号、分号或竖线、`Alice&Bob`、中西文逗号分隔的两个人名、多个逗号，或多个中文姓名由空白分隔。该规则以高召回方式生成候选。 |
| `invalid_separator` | 包含非法分隔符 | 作者名包含竖线、英文/中文分号，或连续两个英文/中文逗号，如 `Alice; Bob`、`张三，，李四`。 |
| `invalid_orcid` | ORCID 不合法 | 非空 ORCID 不符合 `https://orcid.org/0000-0000-0000-000X` 形式，或未通过 MOD 11-2 校验位计算；空字符串允许。 |

## contributors

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `contributors` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值不是字符串列表时标记。 |
| `invalid_separator` | 包含非法分隔符 | 若姓名包含非法分隔符，则标记。 |

## locations

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `locations` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查值是否为列表，并逐项检查列表项类型。 |
| `missing_key` | key 缺失 | 若位置对象缺少必需字段，则标记。 |
| `invalid_value` | 值不合法 | 逐项检查位置对象：`type` 共 4 个限定值，`license` 共 27 个限定值，`is_oa` 共 3 个限定值；例如 `type=download`、`license=cc-by`、`is_oa=true`。 |
| `invalid_url` | URL 不合法 | 若URL 格式错误，则标记。 |

## access_is_oa

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_is_oa` 字段。 |
| `empty` | 值为空 | 若论文开放获取标记为空，则标记。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 3 个限定值之一：`true`、`false`、`unknown`。 |

## access_oa_status

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_oa_status` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 7 个限定值之一，例如 `diamond`、`gold`、`green`、`closed`，也允许空字符串。 |

## access_oa_url

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_oa_url` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值不是字符串列表时标记。 |
| `invalid_url` | URL 不合法 | 若列表中存在无效 URL，则标记。 |

## access_license

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_license` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 27 个限定值之一，例如 `cc-by`、`cc0`、`mit`、`public-domain`，也允许空字符串。 |

## publication_published_date

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_published_date` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `invalid_format` | 格式不符合要求 | 若不符合 YYYY-MM-DD 格式，则标记。 |
| `invalid_date` | 日期不合法 | 若不是有效日历日期，则标记。 |

## publication_published_year

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_published_year` 字段。 |
| `wrong_type` | 字段类型错误 | 值的类型不是整数时标记。 |
| `out_of_range` | 数值超出有效范围 | 若年份超出有效范围，则标记。 |

## publication_venue_issn

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_venue_issn` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值不是字符串列表时标记。 |
| `invalid_format` | 格式不符合要求 | 若ISSN 格式或校验位错误，则标记。 |

## publication_venue_biblio_volume

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_venue_biblio_volume` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `invalid_format` | 格式不符合要求 | 若值不能转换为整数，则标记。 |

## publication_venue_biblio_issue

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_venue_biblio_issue` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `invalid_format` | 格式不符合要求 | 若值不能转换为整数，则标记。 |

## publication_venue_biblio_pages

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_venue_biblio_pages` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `invalid_format` | 格式不符合要求 | 若不符合 `<起始页>-<结束页>` 格式，则标记。 |
| `out_of_range` | 数值超出有效范围 | 若页码不是正数，则标记。 |
| `page_order` | 页码顺序错误 | 若起始页大于结束页，则标记。 |

## publication_pages

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_pages` 字段。 |
| `wrong_type` | 字段类型错误 | 值的类型不是整数时标记。 |
| `out_of_range` | 数值超出有效范围 | 若页数不大于 0，则标记。 |

## publication_venue_name_unified

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `publication_venue_name_unified` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查当前字段及其依赖字段的类型是否符合要求。 |
| `missing_dependency` | 缺少依赖字段 | 若缺少期刊原始名称，无法校验，则标记。 |
| `mismatch` | 值与预期不一致 | 若与预期统一名称不一致，则标记。 |

## grade_class

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `grade_class` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 5 个限定值之一：`k12`、`higher-edu`、`vocational-edu`、`other` 或空字符串。 |

## grade

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `grade` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查当前字段与 `grade_class` 是否均为字符串。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 4 个限定值之一：`小学`、`初中`、`高中` 或空字符串。 |
| `grade_mismatch` | 年级与教育类型不匹配 | 若非 K12 类型设置了年级，则标记。 |

## references

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `references` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查值是否为列表，并逐项检查列表项类型。 |
| `invalid_keys` | key 不符合要求 | 若列表项字段不符合要求，则标记。 |
| `empty` | 值为空 | 若`id_type` 或非 DOI 标识符为空，则标记。 |
| `title_null` | 标题为 null | 检查列表项中的 `title` 是否为 `null`。 |
| `title_wrong_type` | 标题类型错误 | 列表项中的 `title` 不是字符串时标记。 |
| `title_html_tag_layout` | 标题包含 HTML 排版标签 | 列表项的 `title` 包含 HTML 排版标签，如 `<i>`、`<b>`、`<p>`、`<br>`、`<sup>`、`<sub>`、`<em>`、`<strong>`、`<span>`、`<div>`、`<u>`、`<scp>`、`<tt>`。 |
| `title_html_tag_math` | 标题包含 MathML 标签 | 列表项的 `title` 包含 MathML 标签，如 `<math>`、`<mrow>`、`<mi>`、`<mn>`、`<mo>`、`<msub>`、`<mfrac>`、`<msqrt>` 等，也匹配 `mml:` 前缀。 |
| `title_html_tag_xml_comment` | 标题包含 XML 注释 | 列表项的 `title` 包含 XML/HTML 注释片段，如 `<!-- comment -->`。 |
| `title_html_tag_cdata` | 标题包含 CDATA 内容 | 列表项的 `title` 包含 CDATA 区段，如 `<![CDATA[...]]>`。 |
| `title_html_entity_named` | 标题包含命名 HTML 实体 | 列表项的 `title` 包含 `&名称;` 形式的命名 HTML 实体，如 `&nbsp;`、`&amp;`。 |
| `title_html_entity_decimal` | 标题包含十进制 HTML 实体 | 列表项的 `title` 包含 `&#数字;` 形式的十进制 HTML 实体，如 `&#160;`。 |
| `title_html_entity_hex` | 标题包含十六进制 HTML 实体 | 列表项的 `title` 包含 `&#x十六进制;` 形式的 HTML 实体，如 `&#xA0;`。 |
| `title_special_char_invisible` | 标题包含不可见字符 | 列表项的 `title` 包含 U+2000～U+200F、U+202F、U+205F、U+3000、U+FEFF、U+00A0 或 U+2060～U+206F 中的字符。 |
| `title_special_char_replacement` | 标题包含 Unicode 替换字符 | 列表项的 `title` 包含 Unicode 替换字符 `�`（U+FFFD）。 |
| `title_special_char_control` | 标题包含控制字符 | 列表项的 `title` 包含 U+0000～U+0008、U+000B、U+000C、U+000E～U+001F 或 U+007F；TAB、换行和回车除外。 |
| `title_special_char_markup` | 标题包含方括号排版标记 | 列表项的 `title` 包含 `[!i]`、`[!/i]`、`[!sub]`、`[!/sup]`、`[!]`、`[○!R]`、`[○!上]`、`[○!下]` 等标记。 |
| `id_empty` | 标识符为空 | 若DOI 为空，则标记。 |
| `id_wrong_type` | 标识符类型错误 | 当标识符类型为 DOI 时，标识符值不是字符串则标记。 |
| `id_not_lowercase` | 标识符字母大小写不符合要求 | 若DOI 不是小写，则标记。 |
| `id_format_invalid` | 标识符格式不符合要求 | 若DOI 格式错误，则标记。 |
| `id_error_prefix` | 标识符前缀不符合要求 | 当标识符类型为 DOI 时，检查前缀是否命中 3 个测试前缀：`10.0000`、`10.0001`、`10.5555`。 |

## related_works

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `related_works` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查值是否为列表，并逐项检查列表项类型。 |
| `invalid_keys` | key 不符合要求 | 若列表项字段不符合要求，则标记。 |
| `empty` | 值为空 | 若`id_type` 或非 DOI 标识符为空，则标记。 |
| `title_null` | 标题为 null | 检查列表项中的 `title` 是否为 `null`。 |
| `title_wrong_type` | 标题类型错误 | 列表项中的 `title` 不是字符串时标记。 |
| `title_html_tag_layout` | 标题包含 HTML 排版标签 | 列表项的 `title` 包含 HTML 排版标签，如 `<i>`、`<b>`、`<p>`、`<br>`、`<sup>`、`<sub>`、`<em>`、`<strong>`、`<span>`、`<div>`、`<u>`、`<scp>`、`<tt>`。 |
| `title_html_tag_math` | 标题包含 MathML 标签 | 列表项的 `title` 包含 MathML 标签，如 `<math>`、`<mrow>`、`<mi>`、`<mn>`、`<mo>`、`<msub>`、`<mfrac>`、`<msqrt>` 等，也匹配 `mml:` 前缀。 |
| `title_html_tag_xml_comment` | 标题包含 XML 注释 | 列表项的 `title` 包含 XML/HTML 注释片段，如 `<!-- comment -->`。 |
| `title_html_tag_cdata` | 标题包含 CDATA 内容 | 列表项的 `title` 包含 CDATA 区段，如 `<![CDATA[...]]>`。 |
| `title_html_entity_named` | 标题包含命名 HTML 实体 | 列表项的 `title` 包含 `&名称;` 形式的命名 HTML 实体，如 `&nbsp;`、`&amp;`。 |
| `title_html_entity_decimal` | 标题包含十进制 HTML 实体 | 列表项的 `title` 包含 `&#数字;` 形式的十进制 HTML 实体，如 `&#160;`。 |
| `title_html_entity_hex` | 标题包含十六进制 HTML 实体 | 列表项的 `title` 包含 `&#x十六进制;` 形式的 HTML 实体，如 `&#xA0;`。 |
| `title_special_char_invisible` | 标题包含不可见字符 | 列表项的 `title` 包含 U+2000～U+200F、U+202F、U+205F、U+3000、U+FEFF、U+00A0 或 U+2060～U+206F 中的字符。 |
| `title_special_char_replacement` | 标题包含 Unicode 替换字符 | 列表项的 `title` 包含 Unicode 替换字符 `�`（U+FFFD）。 |
| `title_special_char_control` | 标题包含控制字符 | 列表项的 `title` 包含 U+0000～U+0008、U+000B、U+000C、U+000E～U+001F 或 U+007F；TAB、换行和回车除外。 |
| `title_special_char_markup` | 标题包含方括号排版标记 | 列表项的 `title` 包含 `[!i]`、`[!/i]`、`[!sub]`、`[!/sup]`、`[!]`、`[○!R]`、`[○!上]`、`[○!下]` 等标记。 |
| `id_empty` | 标识符为空 | 若DOI 为空，则标记。 |
| `id_wrong_type` | 标识符类型错误 | 当标识符类型为 DOI 时，标识符值不是字符串则标记。 |
| `id_not_lowercase` | 标识符字母大小写不符合要求 | 若DOI 不是小写，则标记。 |
| `id_format_invalid` | 标识符格式不符合要求 | 若DOI 格式错误，则标记。 |
| `id_error_prefix` | 标识符前缀不符合要求 | 当标识符类型为 DOI 时，检查前缀是否命中 3 个测试前缀：`10.0000`、`10.0001`、`10.5555`。 |

## citations

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `citations` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查值是否为列表，并逐项检查列表项类型。 |
| `invalid_keys` | key 不符合要求 | 若列表项字段不符合要求，则标记。 |
| `empty` | 值为空 | 若`id_type` 或非 DOI 标识符为空，则标记。 |
| `title_null` | 标题为 null | 检查列表项中的 `title` 是否为 `null`。 |
| `title_wrong_type` | 标题类型错误 | 列表项中的 `title` 不是字符串时标记。 |
| `title_html_tag_layout` | 标题包含 HTML 排版标签 | 列表项的 `title` 包含 HTML 排版标签，如 `<i>`、`<b>`、`<p>`、`<br>`、`<sup>`、`<sub>`、`<em>`、`<strong>`、`<span>`、`<div>`、`<u>`、`<scp>`、`<tt>`。 |
| `title_html_tag_math` | 标题包含 MathML 标签 | 列表项的 `title` 包含 MathML 标签，如 `<math>`、`<mrow>`、`<mi>`、`<mn>`、`<mo>`、`<msub>`、`<mfrac>`、`<msqrt>` 等，也匹配 `mml:` 前缀。 |
| `title_html_tag_xml_comment` | 标题包含 XML 注释 | 列表项的 `title` 包含 XML/HTML 注释片段，如 `<!-- comment -->`。 |
| `title_html_tag_cdata` | 标题包含 CDATA 内容 | 列表项的 `title` 包含 CDATA 区段，如 `<![CDATA[...]]>`。 |
| `title_html_entity_named` | 标题包含命名 HTML 实体 | 列表项的 `title` 包含 `&名称;` 形式的命名 HTML 实体，如 `&nbsp;`、`&amp;`。 |
| `title_html_entity_decimal` | 标题包含十进制 HTML 实体 | 列表项的 `title` 包含 `&#数字;` 形式的十进制 HTML 实体，如 `&#160;`。 |
| `title_html_entity_hex` | 标题包含十六进制 HTML 实体 | 列表项的 `title` 包含 `&#x十六进制;` 形式的 HTML 实体，如 `&#xA0;`。 |
| `title_special_char_invisible` | 标题包含不可见字符 | 列表项的 `title` 包含 U+2000～U+200F、U+202F、U+205F、U+3000、U+FEFF、U+00A0 或 U+2060～U+206F 中的字符。 |
| `title_special_char_replacement` | 标题包含 Unicode 替换字符 | 列表项的 `title` 包含 Unicode 替换字符 `�`（U+FFFD）。 |
| `title_special_char_control` | 标题包含控制字符 | 列表项的 `title` 包含 U+0000～U+0008、U+000B、U+000C、U+000E～U+001F 或 U+007F；TAB、换行和回车除外。 |
| `title_special_char_markup` | 标题包含方括号排版标记 | 列表项的 `title` 包含 `[!i]`、`[!/i]`、`[!sub]`、`[!/sup]`、`[!]`、`[○!R]`、`[○!上]`、`[○!下]` 等标记。 |
| `id_empty` | 标识符为空 | 若DOI 为空，则标记。 |
| `id_wrong_type` | 标识符类型错误 | 当标识符类型为 DOI 时，标识符值不是字符串则标记。 |
| `id_not_lowercase` | 标识符字母大小写不符合要求 | 若DOI 不是小写，则标记。 |
| `id_format_invalid` | 标识符格式不符合要求 | 若DOI 格式错误，则标记。 |
| `id_error_prefix` | 标识符前缀不符合要求 | 当标识符类型为 DOI 时，检查前缀是否命中 3 个测试前缀：`10.0000`、`10.0001`、`10.5555`。 |

## supplementary_material

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `supplementary_material` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查值是否为列表，并逐项检查列表项及其属性类型。 |
| `invalid_keys` | key 不符合要求 | 若列表项字段不符合要求，则标记。 |

## cited_by_api_url

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `cited_by_api_url` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `invalid_url` | URL 不合法 | 若URL 格式错误，则标记。 |

## access_xinghe_repository_sha256

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_xinghe_repository_sha256` 字段。 |
| `null` | 值为 null | 检查字段值是否为 `null`。 |
| `wrong_type` | 字段类型错误 | 检查当前字段的类型，并检查相关全文状态标记的类型。 |
| `required` | 必填值为空 | `access_xinghe_repository_has_fulltext=true` 时，SHA256 是空字符串或空列表；当前实现不校验十六进制长度或摘要格式。 |

## access_xinghe_repository_origin_path

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_xinghe_repository_origin_path` 字段。 |
| `wrong_type` | 字段类型错误 | 检查当前字段的类型，并检查相关全文状态标记的类型。 |
| `required` | 必填值为空 | 若存在全文时原始路径不能为空，则标记。 |

## access_xinghe_repository_model_name

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_xinghe_repository_model_name` 字段。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `required` | 必填值为空 | 若处理成功时模型名称不能为空，则标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 2 个限定值之一：`mineru`、`llm-web-kit`。 |

## access_xinghe_repository_model_version

| Label 名 | 中文名称 | 标签含义 |
|---|---|---|
| `missing_field` | 字段缺失 | 检查输入对象中是否存在 `access_xinghe_repository_model_version` 字段。 |
| `wrong_type` | 字段类型错误 | 值的类型不是字符串时标记。 |
| `required` | 必填值为空 | 若当前条件下模型版本不能为空，则标记。 |
| `unsupported_value` | 值不在限定范围内 | 值须属于 4 个限定值之一：`1.3.1`、`2`、`2.5`、`4.1.1`。 |
| `model_mismatch` | 模型版本与模型名称不匹配 | `mineru` 只允许 `1.3.1`、`2`、`2.5`，`llm-web-kit` 只允许 `4.1.1`。 |
