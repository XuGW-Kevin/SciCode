import json
import os
import re
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare results of two models.')
    parser.add_argument('A', type=str, help='Name of the first model (e.g., gpt-4o-mini)')
    parser.add_argument('B', type=str, help='Name of the second model (e.g., gpt-4o)')
    return parser.parse_args()

def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_correct_sub_questions(results):
    correct_set = set()
    for question, sub_questions in results.items():
        for sub_question in sub_questions:
            correct_set.add(sub_question)
    return correct_set

def read_last_function_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()

        # Find all function definitions in the file, accounting for default parameters and multiline signatures
        function_regex = r'def [\w_]+\([\w_,=\s\*\.\[\]]*\):'
        functions = list(re.finditer(function_regex, code))

        if not functions:
            return "No function found in the file."

        last_function_start = functions[-1].start()
        last_function_code = code[last_function_start:]
        last_function_code_lines = last_function_code.split('\n')
        last_function_block = []
        in_function_block = False
        for line in last_function_code_lines:
            stripped_line = line.strip()
            if stripped_line.startswith('def '):
                if in_function_block:
                    break
                in_function_block = True
            if in_function_block:
                last_function_block.append(line)

        return "\n".join(last_function_block).strip()
    except FileNotFoundError:
        return "File not found."

def generate_comparison(sub_questions, correct_dir, wrong_dir, correct_label, wrong_label):
    comparisons = []
    for sub_question in sub_questions:
        correct_file_path = os.path.join(correct_dir, f"{sub_question}.py")
        wrong_file_path = os.path.join(wrong_dir, f"{sub_question}.py")

        correct_code = read_last_function_from_file(correct_file_path)
        wrong_code = read_last_function_from_file(wrong_file_path)

        comparison = (
            f"For problem {sub_question}, {correct_label} can solve it correctly but {wrong_label} cannot.\n"
            f"Correct code of {correct_label}:\n{correct_code}\n"
            f"Wrong code of {wrong_label}:\n{wrong_code}\n"
        )
        comparisons.append(comparison)
    return comparisons

def save_comparisons_to_file(comparisons, A, B):
    output_dir = './eval_results/compare/'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'compare_{A}_and_{B}.json')

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(comparisons, output_file, ensure_ascii=False, indent=2)

def main():
    args = parse_arguments()
    A = args.A
    B = args.B

    A_file_path = f'./eval_results/{A}.json'
    B_file_path = f'./eval_results/{B}.json'

    A_results = load_results(A_file_path)
    B_results = load_results(B_file_path)

    A_correct = get_correct_sub_questions(A_results)
    B_correct = get_correct_sub_questions(B_results)

    A_but_not_B = sorted(A_correct - B_correct)
    B_but_not_A = sorted(B_correct - A_correct)

    B_dir = f'./eval_results/generated_code/{B}/'
    A_dir = f'./eval_results/generated_code/{A}/'

    comparisons_B_but_not_A = generate_comparison(B_but_not_A, B_dir, A_dir, B, A)
    comparisons_A_but_not_B = generate_comparison(A_but_not_B, A_dir, B_dir, A, B)

    all_comparisons = comparisons_A_but_not_B # comparisons_B_but_not_A + comparisons_A_but_not_B

    save_comparisons_to_file(all_comparisons, A, B)
    print(f"Comparison results have been saved to './eval_results/compare/compare_{A}_and_{B}.json'")

if __name__ == '__main__':
    main()


# import copy
# from dotenv import load_dotenv
# load_dotenv()
# import textgrad
# from textgrad import EngineLM
# from textgrad import Variable
# from textgrad.engine import get_engine
# from textgrad.optimizer import TextualGradientDescent
# from textgrad.tasks import load_instance_task
# from textgrad.autograd.function import Module
# from textgrad.autograd import FormattedLLMCall

# CODE_INSTANCE_ROLE_DESCRIPTION = "Code generated for scientific problems that must be evaluated for correctness"
# SYSTEM_PROMPT_FOR_FIRST_CODE = """You are a helpful assistant."""
# DEFAULT_TEST_TIME_WITH_TESTS = """You are an intelligent assistant used for evaluating scientific code implementations. Your task is to analyze the code for scientific correctness."""

# import os
# if os.getenv("META_LEARNING_SCRIPT"):
#     META_LEARNING_SCRIPT = os.getenv("META_LEARNING_SCRIPT")
# else:
#     META_LEARNING_SCRIPT = """Provide concise feedback focused solely on the scientific accuracy of the code.
#     Key points to consider:
    
#     1. If you are absolutely certain there are scientific errors in the code, point them out and provide the correct scientific background. If you are not sure, do not suggest any changes regarding scientific errors.
#     2. Check if the calulations in the code have correct formats, dimensions, and signs. If there are errors, point out how to correct them.
#     3. Do not change the input format. If the code adds or removes input variables, suggest removing those changes.
#     4. Avoid feedback on variable names, code style, or efficiency.
#     5. Do not consider whether an input is illegal within data ranges; assume all inputs are valid.
#     6. Only consider different cases if they are clearly indicated by an input variable (e.g., "cut off distance"). If the code handles edge cases not clearly indicated by any input variable, suggest removing those cases. 
    
#     Do not include the given code or provide a revised implementation in your response."""

# DEFAULT_TEST_TIME_WITH_TESTS += META_LEARNING_SCRIPT

# class CodeTestTime(Module):
#     def __init__(self,
#                  engine: EngineLM):
#         super().__init__()
#         tt_system_prompt = DEFAULT_TEST_TIME_WITH_TESTS
#         self.tt_system_prompt = Variable(tt_system_prompt,
#                                             requires_grad=False,
#                                             role_description="system prompt for the evaluation of the code solution")
#         self.engine = engine
#         format_string = "You are a language model that evaluates a python code snippet for solving a scientific problem.\n"
#         format_string += """{{problem}}\n**{role}**{{program}}**\n Investigate the scientific problem and the provided implementation. 
#                         Provide concise feedback focused solely on the scientific accuracy of the code."""
#         format_string += META_LEARNING_SCRIPT
#         self.format_string = format_string.format(role=CODE_INSTANCE_ROLE_DESCRIPTION)
#         self.fields = {"problem": None, "program": None}
#         self.formatted_llm_call = FormattedLLMCall(engine=self.engine,
#                                                    format_string=self.format_string,
#                                                    fields=self.fields,
#                                                    system_prompt=self.tt_system_prompt)

#     def forward(self, problem: str, program: Variable) -> Variable:
#         problem_variable = Variable(problem,
#                                      requires_grad=False,
#                                      role_description="the coding problem")
#         inputs = {"program": program, "problem": problem_variable}
#         return self.formatted_llm_call(inputs=inputs,
#                                        response_role_description=f"evaluation of the {program.get_role_description()}")



# def optimization_one_iteration(optimizer, instance_var, prompt, ENGINE_API):
#     """
#     This is a single iteration of optimization
#     :param optimizer:
#     :param instance_var:
#     :param prompt:
#     :return:
#     """
#     optimizer.zero_grad()
#     loss_fn = CodeTestTime(engine=ENGINE_API)
#     test_time_loss = loss_fn(prompt, instance_var)
#     test_time_loss.backward(engine=ENGINE_API)
#     optimizer.step()
#     return


# def generate_starting_solution(prompt, ENGINE_API):
#     """
#     This is the first attempt at solving the problem.
#     :param prompt:
#     :return:
#     """
#     llm_first_code = ENGINE_API.generate(prompt, system_prompt=SYSTEM_PROMPT_FOR_FIRST_CODE)
#     return llm_first_code

# def generate_textgrad_response(prompt: str, *, model="textgrad-gpt-4-turbo-2024-04-09",
#                              temperature: float = 0) -> str:
#     """
#     :param prompt:
#     :return:
#     """
#     MAX_ITERS = 1
#     model = model[9:]
#     ENGINE_API = get_engine(engine_name=model) # seed
#     generated_programs = []
#     gpt_4_first_code = generate_starting_solution(prompt, ENGINE_API)
#     n_iter = 0
#     generated_programs.append({"code": gpt_4_first_code,
#                                "gradients": None,
#                                "iteration": n_iter,
#                                })

#     instance_var = Variable(gpt_4_first_code, requires_grad=True,
#                             role_description=CODE_INSTANCE_ROLE_DESCRIPTION)

#     optimizer = TextualGradientDescent(engine=ENGINE_API,
#                                        parameters=[instance_var],
#                                        constraints=["Do not add asserts or raise errors to the code",
#                                                     "Do not include dependencies at the beginning of the code",
#                                                     "Do not consider edge cases",
#                                                     "Do not add or remove input variables",
#                                                     "Do not make scientific errors"])

#     for iter in range(MAX_ITERS):
#         optimization_one_iteration(optimizer, instance_var, prompt, ENGINE_API)
#         n_iter += 1
#         generated_programs.append({"code": instance_var.value,
#                                    "gradients": str(instance_var.gradients),
#                                    "iteration": n_iter,
#                                    })

#     return generated_programs[-1]["code"]
