import os
### BASIC FUNCTIONS ###
def generator(generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation, example_directory):
    def generate(instance_text, problem_relaxation="full", cot_type="", n_examples=0, magic="", example_prefix="example_basic-", extraction_label=""):
        # cot_type      :      name of the cot prompt (leave blank for no thought annotation)
        # n_examples    :      number of examples to provide
        # magic         :      "let's think step by step" or whatever appended to end of prompt

        # Just so that we don't double store 0-example CoT prompts on accident:
        if n_examples == 0: cot_type = ""

        # Boilerplate instructions
        instructions = generate_instructions(problem_relaxation)

        # format (stored in data/color_verification) is graph instance with comments appended giving colorings of various types
        if extraction_label not in instance_text: print(f"There is no {extraction_label} key in {instance_text}")
        current_query = generate_query(instance_text, extraction_label)

        prompt = instructions + generate_cot(cot_type, n_examples, magic, example_prefix, example_directory, generate_query, generate_thoughts, generate_correct_evaluation) + current_query + "\n" + magic + "\n[Evaluation]\n"
        return prompt
    return generate

### DATA UTILITIES ###
def load_instance_list(directory, prefix):
    instance_list = []
    for instance in os.listdir(directory):
        if instance.startswith(prefix):
            with open(directory+instance,"r") as fp:
                instance_text = fp.read()
            instance_list.append(instance_text)
    return instance_list

### COT PROMPT UTILITIES ###
def generate_cot(cot_type, n_examples, magic, example_prefix, example_directory, generate_query, generate_thoughts, generate_correct_evaluation):
    # Example instances have to contain a "c example " line with the example coloring
    # TODO make generate_cot depend on instructions_type
    example_instances = load_instance_list(example_directory, example_prefix)

    example_queries = [generate_query(example, "example") for example in example_instances]
    example_thoughts = [generate_thoughts(example, cot_type) for example in example_instances]
    example_evaluations = [generate_correct_evaluation(example, "example") for example in example_instances]

    examples = list(map(lambda x,y,z: x+"\n"+magic+y+"\n"+z+"\n",example_queries, example_thoughts, example_evaluations))
    return "".join(examples[:n_examples])