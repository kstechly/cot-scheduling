import os
import utils
### BASIC FUNCTIONS ###
def generator(domain_name, generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation):
    def generate(instance_text, problem_relaxation="full", cot_type="", n_examples=0, magic=""):
        # cot_type      :      name of the cot prompt (leave blank for no thought annotation)
        # n_examples    :      number of examples to provide
        # magic         :      "let's think step by step" or whatever appended to end of prompt

        # Boilerplate instructions
        instructions = generate_instructions(problem_relaxation)

        # TODO GET RID OF THIS NONSENSE!!!
        current_query = generate_query(instance_text)

        prompt = "[Instructions]\n"+instructions
        if n_examples: prompt+=f"\n\nThe following {n_examples} examples are provided. Please follow the formatting used in them.\n\n"
        prompt += generate_cot(cot_type, n_examples, magic, domain_name, generate_query, generate_thoughts, generate_correct_evaluation, problem_relaxation) + f'\nProblem to solve:\n\n' + current_query + "\n\n" + magic
        if cot_type or magic: prompt+= "\n\n[Thoughts]"
        else: prompt+= "\n[Answer]\n"
        return prompt
    return generate

### COT PROMPT UTILITIES ###
def generate_cot(cot_type, n_examples, magic, domain_name, generate_query, generate_thoughts, generate_correct_evaluation, problem_relaxation):
    # Example instances have to contain a "c example " line with the example coloring
    # TODO this should know its own name. Just make these classes already cmon
    example_instances = utils.read_json(domain_name, False, "examples")

    assert n_examples <= len(example_instances)

    example_labels = [f'Example {k}:\n\n' for k in example_instances]
    example_queries = [generate_query(example) for example in example_instances.values()]
    example_thoughts = [generate_thoughts(example, cot_type) for example in example_instances.values()]
    example_evaluations = [generate_correct_evaluation(example, problem_relaxation) for example in example_instances.values()]

    examples = list(map(lambda w,x,y,z: w+x+"\n\n"+magic+f"{chr(10)+'[Thoughts]' if cot_type else ''}\n"+y+"\n\n[Answer]\n"+z+"\n\n",example_labels, example_queries, example_thoughts, example_evaluations))
    return "".join(examples[:n_examples])