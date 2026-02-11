from transformers import AutoTokenizer

class Prompter:#，回答案例字数控制在500字以内
    def __init__(self,tokenizer_path="/data/jcxy/llm_model/Qwen2-7B-Instruct-AWQ"):

        self.PAIRWISE_COMPARE_PROMPT = """你是一位擅长评分的专家。根据给定的评分标准，评估两个助手的回答。
###给定问题
{prompt}
##先验经验##
{prior_experience}
###助手设定1
{system_1}
###助手回答1 
[回答开始] {response_1} [回答结束]
—
###助手设定2
{system_2}
###助手回答2 
[回答开始] {response_2} [回答结束]
—
###输出要求 
输出七个部分：
1. 特有评估标准：列出针对当前用户问题的特有评估标准，每个维度的评分标准。
2. 分析：根据通用评估标准和特有评估标准，结合上面的思考和答案中的结果，对比并分析两个助手的回答，重点分析每个维度的表现，让我们一步一步的思考。
3. 权重分配：列出通用评估标准和特有评估标准的权重分配，确保总权重为 100%。
4. 打分：计算两个助手回答的每个评估维度的得分；然后计算加权平均得分，计算过程使用数学公式，不要含有 Markdown。
5. 输出最终得分：输出格式为 \\boxed{{分数1,分数2}}，得分之间以逗号分隔。
6. 新经验：基于上述分析生成多个独立、不重复的，凝练地总结出核心问题和关键的改进原则或经验教训。这是从具体问题到通用解决方案的提炼。
7.系统提示词进化：
 - 分析两个回复在 System Prompt 执行层面上的根本差异。
 - 提取胜出者的关键成功因子，吸取落败者的教训。
 - 系统提示词进化：分析胜出者的关键成功因子和落败者的教训。基于此分析，生成一个全新的、完整的、更强的可以直接使用的System Prompt。
"""

        self.breadth_base_instruction = """我希望你充当一个提示创造者。
你的目标是从#给定的提示#中汲取灵感，创造一个全新的提示,全新的提示设定不能包含与ai助手相关的内容。
这个新提示应该属于与#给定的提示#相同的领域，但要更加独特。
#创造的提示#的长度和复杂性应与#给定的提示#相似。
#创造的提示#必须合理，并且必须能被人理解和回应。
'给定的提示'和'创造的提示'不允许出现在#创造的提示#中。
#创造的提示#应当跟#给定的提示#为同一种语言。
避免使用诸如“创造的提示如下：”等引导性语句
'#给定的提示#':{instruction}
'#创造的提示#':"""

        self.Create_Expert_Prompt = """
对于每一条指令，请以第二人称视角，撰写一段高质量的描述，说明最适合回答该指令的角色。
语言一致性：例如[指令]为英文，[角色描述]就也为英文。
内容独特性：不包含对[指令]文本的任何复制或照搬。

[指令]：{question}  
[角色描述]："""

        self.depth_base_instruction = """我希望你充当一个提示重写者。
你的目标是将给定的提示重写为一个更复杂的版本，以使那些著名的人工智能系统（如ChatGPT和GPT-4）更难处理,全新的提示设定不能包含与ai助手相关的内容。。
但重写的提示必须合理，并且必须能被人理解和回应。
你的重写不能省略#给定的提示#中的非文本部分，例如表格和代码。此外，请不要省略#给定的提示#中的输入。
你应该使用以下方法来复杂化给定的提示：
{method} 
你应尽力不使#重写的提示#变得冗长，#重写的提示#只能在#给定的提示#中增加10到20个单词。
'给定的提示'和'重写的提示'不允许出现在#重写的提示#中。
#重写的提示#应当跟#给定的提示#为同一种语言。
避免使用诸如“重写的提示如下：”等引导性语句
'#给定的提示#':{instruction}
'#重写的提示#':"""

        self.system_breadth_base_instruction = """我希望你充当一个角色扮演提示词创造者，系统提示内容应属于角色扮演内容或者与#给定提示#内容不是强包含关系的内容以避免与给定提示内容重复。
你的目标是从#给定提示#中汲取灵感，创造一个全新的角色扮演提示词,全新的提示设定不能包含与ai助手相关的内容。
#创造的提示#必须合理，并且必须能被人理解和回应。
'给定提示'和'创造的提示'不允许出现在#创造的提示#中。
避免使用诸如“创造的提示如下：”等引导性语句
'#给定提示#':{system}
'#创造的提示词#':"""

        self.system_depth_base_instruction = """
我希望你充当一个角色扮演提示词重写者。
你的目标是基于#给定的提示#，在其合理性和可理解性的前提下，对该提示词进行复杂化、纠错和优化。
你需要创造一个新的角色扮演提示词（#重写的提示词#）：
你应该使用以下方法来复杂化#给定的提示#：
{method}
请注意：
1. '给定的提示'和'重写的提示词'这几个短语不允许出现在#重写的提示词#中。
2. 提示词依然必须可理解、无语法错误、逻辑自洽。
3. 系统提示内容应属于人物设定内容。
4. 避免使用诸如"重写的提示词如下："等引导性语句
'#给定提示#':{system}
'#重写的提示词#':
"""
        self.OPTIMIZE_EXPERIENCE_PROMPT = """
# 任务：优化智能体经验库

你是一个经验库优化专家。你的任务是审查一批新生成的"现有经验"，并结合"过往经验"，制定一个最终的经验库修订计划。

你的目标是维护一个高质量、简洁、通用且无冗余的、能够服务于未来所有可能任务的经验库。

## ⚠️ 核心原则：优先保留，谨慎删除

删除操作风险很高，可能导致知识丢失。一条过往经验不应仅仅因为它与当前的“现有经验”不直接相关就被删除。 经验库是为未来所有未知任务服务的，而不是只为当前任务。

把经验库想象成一个工具箱：一个特殊的工具（一条具体的过往经验）并不会因为当前的工作需要另一个工具而变得无用。它对于未来的特定工作仍然至关重要。

## 修订计划操作指南

你可以使用以下四种操作来制定计划：

1.  "add": 如果一条"现有经验"是全新的、有价值的且与过往经验不重复，则将其添加。
2.  "modify": 如果一条"现有经验"可以显著改进或泛化一条"过往经验"，则用新内容替换旧的。
3.  "merge": 如果多条经验表达了相似的思想，将它们合并成一条更通用、更精炼的新经验，并删除被合并的旧经验。
4.  "delete": 仅当一条"过往经验"满足以下严格条件之一时，才将其删除：
    *   事实性错误: 该经验被证明是错误的或导向一个有害的策略。
    *   完全冗余: 该经验的全部思想被另一条更优、更通用的经验完全覆盖。如果它仍有独特的价值或适用场景，就不应删除。
    *   质量极低: 该经验的表述极其模糊、毫无指导意义。

## 经验质量标准

1.  清晰通用: 经验应该是普适的指导原则，而不是针对某个具体问题的解决方案。
2.  聚焦策略: 强调战略性思维模式，而非具体计算。
3.  避免重复: 确保经验库中没有语义上重复的条目。

## 输入

### 过往经验
{existing_experiences}

### 现有经验
{now_experiences}

## 你的任务

请仔细分析上述输入，并按照以下JSON格式返回你的最终修订计划。一步一步思考，在你的思考过程中，请明确论证你每一次"delete"或"merge"操作的合理性，确保它符合上述的严格标准。

```json
{{
  "think": "在这里进行你的思考过程，详细说明每一步决策的依据，特别是删除和合并操作...",
  "plan": [
    {{
      "option": "add",
      "experience": "要添加的新经验内容。"
    }},
    {{
      "option": "modify",
      "modified_from_id": "要修改的现有经验的ID",
      "experience": "修改后的新经验内容。"
    }},
    {{
      "option": "merge",
      "merged_from_ids": ["要合并的经验ID_1", "要合并的经验ID_2"],
      "experience": "合并后的新经验内容。"
    }},
    {{
      "option": "delete",
      "delete_id": "要删除的现有经验的ID。"
    }}
  ]
}}
"""
        # ============================================================================
        # LLaMA-Berry 论文标准提示词模板
        # 参考: LLaMA-Berry - Pairwise Self-Refine with Monte Carlo Tree Search
        # ============================================================================

        # 1. PPRM (Pairwise Preference Reward Model) 提示词
        # 用于成对比较，判断哪个答案更好
        self.LlamaBerry_PPRM_Prompt = """Problem: {problem}

First Answer: {first_answer}

Second Answer: {second_answer}

Is First Answer better than Second Answer?

Please analyze both answers carefully and provide your judgment. Consider:
- Correctness and accuracy
- Completeness of the solution
- Clarity of explanation
- Mathematical rigor

Your answer should be "Yes" or "No" followed by a brief explanation."""

        # 2. Rewriting 操作提示词
        # 用于根据批评生成改进的答案
        self.LlamaBerry_Rewrite_Prompt = """Please write a better answer for this question refer to the comments.

Problem: {problem}

Comments/Critique: {critique}

Your improved answer:
"""

        # 3. Generate Critique 操作提示词
        # 用于生成批评和改进建议
        self.LlamaBerry_Critique_Prompt = """Analyze this weak Answer, write a strict Critic/Reflection for error re-correct and Hints/Guidelines for maximum improvement.

Let's think step by step.

Problem: {problem}

Weak Answer: {weak_answer}

Your critique (including error analysis and improvement guidelines):
"""