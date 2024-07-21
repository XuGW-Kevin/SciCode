CODE_INSTANCE_ROLE_DESCRIPTION = "Code generated that must be evaluated for correctness"
SYSTEM_PROMPT_FOR_FIRST_CODE = """You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).
Use a Python code block to write your response. For example:
```python
print('Hello world!')
```"""

import copy
from dotenv import load_dotenv
load_dotenv()
import textgrad
from textgrad import EngineLM
from textgrad import Variable
from textgrad.engine import get_engine
from textgrad.optimizer import TextualGradientDescent
from textgrad.tasks import load_instance_task
from textgrad.autograd.function import Module
from textgrad.autograd import FormattedLLMCall

DEFAULT_TEST_TIME_WITH_TESTS = """You are an intelligent assistant used as an evaluator, and part of an optimization system. 
You will analyze a code implementation for a coding solution for scientific problems. 
Think about the correctness of the code in various test cases.
Give very concise feedback.
Investigate the code problem and the provided implementation. 
Do not provide a revised implementation. 
If you think the code is incorrect, carefully suggest why there are issues with the code and provide feedback.
"""

default_instruction_test = (f"")

class CodeTestTime(Module):
    def __init__(self,
                 engine: EngineLM,
                 evaluation_instruction: str = default_instruction_test,
                 system_prompt: Variable = None):
        super().__init__()
        if system_prompt:
            self.tt_system_prompt = system_prompt
        else:
            tt_system_prompt = DEFAULT_TEST_TIME_WITH_TESTS
            self.tt_system_prompt = Variable(tt_system_prompt,
                                                requires_grad=False,
                                                role_description="system prompt for the evaluation of the code solution")
        self.engine = engine
        format_string = "You are a language model that evaluates a python code snippet.\n"
        format_string += "{instruction}\n**The coding problem:**\n\n{{problem}}\n**{role}**{{program}}**\n"
        self.format_string = format_string.format(instruction=evaluation_instruction, role=CODE_INSTANCE_ROLE_DESCRIPTION)
        self.fields = {"problem": None, "program": None}
        self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
                                                   format_string=self.format_string,
                                                   fields=self.fields,
                                                   system_prompt=self.tt_system_prompt)

    def forward(self, problem: str, program: Variable) -> Variable:
        problem_variable = Variable(problem,
                                     requires_grad=False,
                                     role_description="the coding problem")
        inputs = {"program": program, "problem": problem_variable}
        return self.formatted_llm_call(inputs=inputs,
                                       response_role_description=f"evaluation of the {program.get_role_description()}")



def optimization_one_iteration(optimizer, instance_var, prompt, ENGINE_API):
    """
    This is a single iteration of optimization
    :param optimizer:
    :param instance_var:
    :param prompt:
    :return:
    """
    optimizer.zero_grad()
    loss_fn = CodeTestTime(engine=ENGINE_API)
    test_time_loss = loss_fn(prompt, instance_var)
    test_time_loss.backward(engine=ENGINE_API)
    optimizer.step()
    return


def generate_starting_solution(prompt, ENGINE_API):
    """
    This is the first attempt at solving the problem.
    :param prompt:
    :return:
    """
    llm_first_code = ENGINE_API.generate(prompt, system_prompt=SYSTEM_PROMPT_FOR_FIRST_CODE)
    return llm_first_code

def generate_textgrad_response(prompt: str, *, model="textgrad-gpt-4-turbo-2024-04-09",
                             temperature: float = 0) -> str:
    """
    :param prompt:
    :return:
    """
    MAX_ITERS = 3
    model = model[9:]
    ENGINE_API = get_engine(engine_name=model) # seed
    generated_programs = []
    gpt_4_first_code = generate_starting_solution(prompt, ENGINE_API)
    n_iter = 0
    generated_programs.append({"code": gpt_4_first_code,
                               "gradients": None,
                               "iteration": n_iter,
                               })

    instance_var = Variable(gpt_4_first_code, requires_grad=True,
                            role_description=CODE_INSTANCE_ROLE_DESCRIPTION)

    optimizer = TextualGradientDescent(engine=ENGINE_API,
                                       parameters=[instance_var],
                                       constraints=["Do not add asserts to the code",
                                                    "Code must contain imports"])

    for iter in range(1 + MAX_ITERS):
        optimization_one_iteration(optimizer, instance_var, prompt, ENGINE_API)
        n_iter += 1
        generated_programs.append({"code": instance_var.value,
                                   "gradients": str(instance_var.gradients),
                                   "iteration": n_iter,
                                   })

    return generated_programs[-1]["code"]
