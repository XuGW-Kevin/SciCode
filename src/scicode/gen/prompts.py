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

CODE_INSTANCE_ROLE_DESCRIPTION = "Code generated for scientific problems that must be evaluated for correctness"
SYSTEM_PROMPT_FOR_FIRST_CODE = """You are a helpful assistant."""
DEFAULT_TEST_TIME_WITH_TESTS = """You are an intelligent assistant used for evaluating scientific code implementations. Your task is to analyze the code for scientific correctness.

Provide concise feedback focused solely on the scientific accuracy of the code.
Key points to consider:

1. If you are absolutely certain there are scientific errors in the code, point them out and provide the correct scientific background. If you are not sure, do not suggest any changes regarding scientific errors.
2. Check if the calulations in the code have correct formats, dimensions, and signs. If there are errors, point out how to correct them.
3. Do not change the input format. If the code adds or removes input variables, suggest removing those changes.
4. Avoid feedback on variable names, code style, or efficiency.
5. Do not consider whether an input is illegal within data ranges; assume all inputs are valid.
6. Only consider different cases if they are clearly indicated by an input variable (e.g., "cut off distance"). If the code handles edge cases not clearly indicated by any input variable, suggest removing those cases. 

Do not include the given code or provide a revised implementation in your response.
"""

class CodeTestTime(Module):
    def __init__(self,
                 engine: EngineLM):
        super().__init__()
        tt_system_prompt = DEFAULT_TEST_TIME_WITH_TESTS
        self.tt_system_prompt = Variable(tt_system_prompt,
                                            requires_grad=False,
                                            role_description="system prompt for the evaluation of the code solution")
        self.engine = engine
        format_string = "You are a language model that evaluates a python code snippet for solving a scientific problem.\n"
        format_string += """{{problem}}\n**{role}**{{program}}**\n Investigate the scientific problem and the provided implementation. 
                        Provide concise feedback focused solely on the scientific accuracy of the code.
                        Key points to consider:

                        1. If you are absolutely certain there are scientific errors in the code, point them out and provide the correct scientific background. If you are not sure, do not suggest any changes regarding scientific errors.
                        2. Check if the calulations in the code have correct formats, dimensions, and signs. If there are errors, point out how to correct them.
                        3. Do not change the input format. If the code adds or removes input variables, suggest removing those changes.
                        4. Avoid feedback on variable names, code style, or efficiency.
                        5. Do not consider whether an input is illegal within data ranges; assume all inputs are valid.
                        6. Only consider different cases if they are clearly indicated by an input variable (e.g., "cut off distance"). If the code handles edge cases not clearly indicated by any input variable, suggest removing those cases. 

                        Do not include the given code or provide a revised implementation in your response."""
        self.format_string = format_string.format(role=CODE_INSTANCE_ROLE_DESCRIPTION)
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
    print("------BEGIN------")
    print(test_time_loss)
    print("------END------")
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
    MAX_ITERS = 2
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
                                       constraints=["Do not add asserts or raise errors to the code",
                                                    "Do not include dependencies at the beginning of the code",
                                                    "Do not consider edge cases",
                                                    "Do not add or remove input variables",
                                                    "Do not make scientific errors"])

    for iter in range(MAX_ITERS):
        optimization_one_iteration(optimizer, instance_var, prompt, ENGINE_API)
        n_iter += 1
        generated_programs.append({"code": instance_var.value,
                                   "gradients": str(instance_var.gradients),
                                   "iteration": n_iter,
                                   })

    return generated_programs[-1]["code"]
