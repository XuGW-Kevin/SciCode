import os
import subprocess
import json
import textgrad
from textgrad import EngineLM
from textgrad import Variable
from textgrad.engine import get_engine
from textgrad.optimizer import TextualGradientDescent
from textgrad.tasks import load_instance_task
from textgrad.autograd.function import Module
from textgrad.autograd import FormattedLLMCall

GPT_MODEL = "gpt-4o-mini"
META_LEARNING_SCRIPT_HEAD = "Provide concise feedback focused solely on the scientific accuracy of the code."
META_LEARNING_SCRIPT_TAIL = "Do not include the given code or provide a revised implementation in your response."
META_LEARNING_SCRIPT = META_LEARNING_SCRIPT_HEAD + "\n" + META_LEARNING_SCRIPT_TAIL
os.environ['META_LEARNING_SCRIPT'] = META_LEARNING_SCRIPT
CODE_INSTANCE_ROLE_DESCRIPTION = "Prompts for evaluation of code for scientific problems that must be improved, especially by reasonably adding, modifying, or deleting key points"

class MetaLearner(Module):
    def __init__(self, engine: EngineLM):
        super().__init__()
        tt_system_prompt = "You are an intelligent assistant used for improving prompts for evaluating code solutions. Yout task is to write better prompts that can help evaluate scientific accuracy of code."
        self.tt_system_prompt = Variable(tt_system_prompt,
                                            requires_grad=False,
                                            role_description="system prompt for the assistant trying to improve prompts for evaluation of code solution")
        self.engine = engine
        format_string = """You are a language model that improves prompts for evaluating python code snippets for solving scientific problems.
                            You are given some code snippets:"""
                            
        format_string += """{{comparison}}\n**{role}**\n {{prompt}} 
        Investigate the difference between the provided code snippets. Provide very concise feedback on how to improve the prompt so that it can better catch errors in wrong code snippets.
        Your suggestions should be as general as possible, focusing on adding, modifying, or deleting key points about how to evaluate code for scientific problems.
        Your suggestions should not be limited to specific code snippets provided.
        For example, you can suggest adding 'Do not change the input format' instead of 'Do not add cut off distance as an input variable'.
        If the prompt is already too long (more than 8 suggestions), try to make it more concise or modify/remove some suggestions before integrating new ones.
        Suggest the prompt to never give feedback on variable names, code style, documentation, data ranges, edge cases or efficiency because we do not care about these in our setting.
        """
        self.format_string = format_string.format(role=CODE_INSTANCE_ROLE_DESCRIPTION)
        self.fields = {"prompt": None, "comparison": None}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.tt_system_prompt)

    def forward(self, comparison: str, prompt: Variable) -> Variable:
        comparison_variable = Variable(comparison,
                                     requires_grad=False,
                                     role_description="the comparison of the code")
        inputs = {"prompt": prompt, "comparison": comparison_variable}
        return self.formatted_llm_call(inputs=inputs,
                                       response_role_description=f"evaluation of the {prompt.get_role_description()}")

def optimization_one_iteration(instance_var, comparison):
    loss_fn = MetaLearner(engine=ENGINE_API)
    test_time_loss = loss_fn(comparison, instance_var)
    test_time_loss.backward(engine=ENGINE_API)
    return

def get_score(model_name):
    file_path = f'./eval_results/{model_name}.txt'
    with open(file_path, 'r') as file:
        content = file.read()
    return int(re.search(r'correct steps:\s+(\d+)/\d+', content).group(1))
    
subprocess.run(["python", "eval/scripts/gencode_json.py", "--model", f"{GPT_MODEL}"])
subprocess.run(["python", "eval/scripts/test_generated_code.py", "--model", f"{GPT_MODEL}"])

OPTIMIZE_STEPS = 100
META_LEARNING_SCRIPT_MIDDLE = """Key points to consider:
    1. KEY_POINT_HERE
    """
PREVIOUS_META_LEARNING_SCRIPT_MIDDLE = META_LEARNING_SCRIPT_MIDDLE
best_score = get_score(GPT_MODEL)

for cycle_number in range(OPTIMIZE_STEPS):
    model_name = f"textgrad*{GPT_MODEL}".replace('*', str(cycle_number))
    
    os.environ['META_LEARNING_SCRIPT'] = META_LEARNING_SCRIPT_HEAD + "\n" + META_LEARNING_SCRIPT_MIDDLE + "\n" + META_LEARNING_SCRIPT_TAIL
    subprocess.run(["python", "eval/scripts/gencode_json.py", "--model", model_name])
    subprocess.run(["python", "eval/scripts/test_generated_code.py", "--model", model_name])
    subprocess.run(["python", "eval/scripts/compare_results.py", f"{GPT_MODEL}", model_name])
    compare_file = f"./eval_results/compare/compare_{GPT_MODEL}_and_{model_name}.json"
    
    current_score = get_score(model_name)
    if current_score > best_score:
        best_score = current_score
        PREVIOUS_META_LEARNING_SCRIPT_MIDDLE = META_LEARNING_SCRIPT_MIDDLE
    else:
        META_LEARNING_SCRIPT_MIDDLE = PREVIOUS_META_LEARNING_SCRIPT_MIDDLE

    with open(compare_file, 'r') as file:
        compare_results = json.load(file)
    
    ENGINE_API = get_engine(engine_name=GPT_MODEL)
    instance_var = Variable(META_LEARNING_SCRIPT_MIDDLE, requires_grad=True,
                            role_description=CODE_INSTANCE_ROLE_DESCRIPTION)
    optimizer = TextualGradientDescent(engine=ENGINE_API,
                                       parameters=[instance_var],
                                       constraints=["list key points using numbers",
                                                    "keep the number of key points less than 8"])
    optimizer.zero_grad()
    result = ""
    for aresult in compare_results:
        result += aresult + "\n"
    optimization_one_iteration(instance_var, result)
    optimizer.step()
    META_LEARNING_SCRIPT_MIDDLE = instance_var.value
    with open("./eval_results/meta_learning_script.txt", "a") as file:
        file.write(f"Cycle {cycle_number}:\n")
        file.write(META_LEARNING_SCRIPT_MIDDLE + "\n")