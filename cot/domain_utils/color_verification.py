import json
from domain_utils import domain

DOMAIN_NAME = "color_verification"
EXAMPLE_DIRECTORY = f"data/examples/{DOMAIN_NAME}/"

def file_ending():
    return ".col"
def generate(*args, **kwargs):
    return domain.generator(generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation, EXAMPLE_DIRECTORY)(*args, **kwargs)
def extraction_labels():
    return ['correct','ablated','non-optimal','random','llm']

def evaluate(instance_text, response_trace, extraction_label="", backprompt_type="", cot_type=""):
    evaluation = {}
    summary = {}
    instance_info = check_instance_info(instance_text, extraction_label)

    coloring_text = extract_coloring(instance_text, extraction_label)
    valid_coloring, minimal_coloring, errors = check_coloring(coloring_text, instance_text.split('c correct')[0])
    evaluation["ground truth"] = {"valid_coloring":valid_coloring, "minimal_coloring":minimal_coloring, "errors":errors}
    ground_truth = valid_coloring and minimal_coloring



    # TODO


    raise NotImplementedError
    # TODO change the output format post thought
    # claim = STOP_PHRASE in response_trace["responses"][-1]
    #summary["binary"] = claim == ground_truth



    
    evaluation["summary"] = summary
    return evaluation

### HELPER FUNCTIONS ###

## GRAPH UTILITIES ##
CHROMATIC_NUMBER_KEY = "OPTIMAL CHROMATIC NUMBER === "

def parse_dimacs(instance_text):
    return [[v for v in line.split()[1:]] for line in instance_text.split("\n") if line.startswith("e")]

def optimal_coloring_number(instance_text):
    return instance_text.split(CHROMATIC_NUMBER_KEY)[1].split("\n")[0]

def parse_coloring(coloring_text):
    coloring = {}
    for line in coloring_text.split("\n"):
        assignment = line.strip().split(": ")
        if len(assignment) < 2: continue #throw out lines that aren't part of the coloring
        coloring[assignment[0]] = assignment[1]
    return coloring

def check_coloring(coloring_text, instance_text):
    # missing_vertices      : list of vertices not mentioned in the coloring
    # wrong_edges           : list of (sorted) lists of edges where both vertices are the same color 
    errors = {"missing_vertices":[], "wrong_edges":[]}
    valid_coloring = True
    minimal_coloring = True
    
    # parse coloring_text into a dictionary with entries like vertex: color
    coloring = parse_coloring(coloring_text)
    
    # check if coloring is valid
    edges = parse_dimacs(instance_text)
    for edge in edges:
        if edge[0] not in coloring:
            valid_coloring = False 
            errors["missing_vertices"].append(edge[0])
        if edge[1] not in coloring:
            valid_coloring = False 
            errors["missing_vertices"].append(edge[1])
        elif edge[0] in coloring and coloring[edge[0]] == coloring[edge[1]]:
            valid_coloring = False
            errors["wrong_edges"].append(sorted([edge[0], edge[1]])) 
    
    # check if coloring is optimal
    if int(optimal_coloring_number(instance_text)) < len(set(coloring.values())): minimal_coloring = False
    
    return valid_coloring, minimal_coloring, errors

## DATA UTILITIES ##

def check_instance_info(instance_text, extraction_label):
    
    #TODO 
    # idea: print out the size of the instance etc
    
    raise NotImplementedError


    prompt = ""
    num_verts = 0
    min_vert = 1
    for edge in parse_dimacs(instance_text):
        prompt += f"Vertex {edge[0]} is connected to vertex {edge[1]}.\n"
        num_verts = max(num_verts, int(edge[0]),int(edge[1]))
        min_vert = min(min_vert, int(edge[0]), int(edge[1]))
    num_verts += (min_vert+1)%2
    return num_verts, prompt

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation == "full":
        instructions = f"[Instructions]\n"
        instructions+= "The graph coloring problem is solved by labeling a graph such that no adjacent (edge-connected) vertices have the same label. You will be given a graph and a proposed coloring. Your task is to verify if the proposed coloring is a solution to the graph coloring problem."
        instructions+= "\nGiven a description of a graph and a coloring for that graph, please evaluate whether the coloring is valid, minimal, and correct or if it failed to color certain vertices or colored two vertices the same along an edge. Provide your answer in JSON format starting on a new line, as described below:\n"
        instructions+= '{\n    "missing_vertices": [],\n    "wrong_edges": [],\n    "valid": true,\n    "minimal": true,\n    "correct": true\n}'
        instructions+= '\nThe missing_vertices list should contain only vertices that are in the graph but not in the coloring. This may be empty.\n'
        instructions+= 'The wrong_edges list should contain tuples of vertices which have an edge between them in the graph but are both colored the same in the coloring.\n'
        instructions+= 'The "valid" boolean should be True if there are no missing vertices and no wrong edges, False otherwise.\n'
        instructions+= f'The "minimal" boolean should be True if the number of colors used in the coloring is the same or smaller than the provided optimal coloring number, but False otherwise'
        instructions+= 'The "correct" boolean should be True if the coloring is both "valid" and "minimal", but False otherwise.'
        instructions+= 'Ensure the JSON part of your answer can be parsed properly by the Python JSON parser, and starts on its own line, right after the [Evaluation] tag.\n\n'
    else: raise NotImplementedError
    return instructions

def generate_query(instance_text, extraction_label):
    # Text Tags
    #  Prompt
    #   [Instructions]
    #   [Graph]
    #   [Coloring]
    #  Output/Examples
    #   [Reasoning]
    #   [Evaluation]
    query = f"[Graph]\n"
    query+= f"The following graph, described as a list of edges, has an optimal coloring number of {optimal_coloring_number(instance_text)}:\n"
    query+= extract_graph(instance_text)
    query+= f"\n[Coloring]\n"
    query+= extract_coloring(instance_text, extraction_label)
    return query

def extract_coloring(instance_text, extraction_label):
    return instance_text.split(f"c {extraction_label}")[1].split("\n")[0].replace("\\n","\n")

def extract_graph(instance_text):
    prompt = ""
    for edge in parse_dimacs(instance_text): prompt += f"Vertex {edge[0]} is connected to vertex {edge[1]}.\n"
    return prompt

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type):
    explanation = "To check if a coloring is valid, it suffices to iterate through every edge in the graph and check 1) if the first vertex has been colored, 2) if the second vertex has been colored, 3) if both vertices are different colors. To check if it is minimal, all we have to do is to count the number of colors and ensure that this number is less than or equal to the optimal coloring number.\n"

    cot = ""
    edges = parse_dimacs(example_instance)
    coloring_text = extract_coloring(example_instance, "example")
    valid_coloring, minimal_coloring, errors = check_coloring(coloring_text, example_instance)
    coloring = parse_coloring(coloring_text)

    if cot_type:
        cot +="\n[Reasoning]\n"
    if not cot_type: pass
    elif cot_type == "global":
        cot += explanation

        # validity check
        cot += "First, we check if the coloring is valid.\n"
        for edge_num in range(0,len(edges)):
            cot += f'Edge {edge_num}:\n'
            cot += f'Edge number {edge_num} is defined in the graph description as an edge between vertex {edges[edge_num][0]} and vertex {edges[edge_num][1]}.\n'
            cot += f'We look at the coloring to see if it mentions the first vertex and see that '
            # TODO: maybe an even more global prompt that checks every single vertex individually manually, doing the full loop
            if edges[edge_num][0] not in errors["missing_vertices"]:
                cot += f'the coloring labels vertex {edges[edge_num][0]} as {coloring[edges[edge_num][0]]}.\n'
            else:
                cot += f'the coloring does not mention vertex {edges[edge_num][0]}. Therefore the coloring is invalid. We keep track of this for later.\nSince there is no defined color, we won\'t compare vertex colors on this edge.\n'
            cot += f'We look at the coloring to see if it mentions the second vertex and see that '
            if edges[edge_num][0] not in errors["missing_vertices"]:
                cot += f'the coloring labels vertex {edges[edge_num][1]} as {coloring[edges[edge_num][1]]}.\n'
            else:
                cot += f'the coloring does not mention vertex {edges[edge_num][1]}. Therefore the coloring is invalid. We keep track of this for later.\nSince there is no defined color, we won\'t compare vertex colors on this edge.\n'
            if edges[edge_num][0] not in errors["missing_vertices"] and edges[edge_num][0] not in errors["missing_vertices"]:
                cot += f'Since both vertices are colored, we can compare them.\n'
                if sorted([edges[edge_num][0],edges[edge_num][1]]) in errors["wrong_edges"]:
                    assert coloring[edges[edge_num][0]] == coloring[edges[edge_num][1]]
                    cot += f'Both vertices are colored {coloring[edges[edge_num][0]]}. Therefore the coloring is invalid. We keep track of this for later.\n'
                else:
                    assert coloring[edges[edge_num][0]] != coloring[edges[edge_num][1]]
                    cot += f'Vertex {edges[edge_num][0]} is colored {coloring[edges[edge_num][0]]}, and vertex {edges[edge_num][1]} is colored {coloring[edges[edge_num][1]]}, which are different colors.\n'

        # minimality check
        cot += f'Now we check if the coloring is minimal. The colors listed are '
        # TODO an even looser version where it iterates through literally everything.
        colors = list(map(str,set(coloring.values())))
        cot += f'{", ".join(colors[:-1])} and {colors[-1]}.'
        cot += f'This is a total of {len(colors)} colors.\n'
        if minimal_coloring:
            assert len(colors) <= int(optimal_coloring_number(example_instance))
            cot += f'{len(colors)} is less than or equal to the optimal coloring number {optimal_coloring_number(example_instance)}, which is minimal.\n'
        else:
            assert len(colors) > optimal_coloring_number(example_instance)
            cot += f'{len(colors)} greater than the optimal coloring number {optimal_coloring_number(example_instance)}, which is not minimal. Therefore the coloring is not correct.\n'
        cot += f'Using all the information we\'ve compiled, we can now write down the final answer.'
    else: raise NotImplementedError
    return cot

def generate_correct_evaluation(example_instance, extraction_label, problem_relaxation):
    if problem_relaxation == "full":
        valid_coloring, minimal_coloring, errors = check_coloring(extract_coloring(example_instance, extraction_label),example_instance)
        evaluation = {"missing_vertices":errors["missing_vertices"],"wrong_edges":errors["wrong_edges"],"valid":valid_coloring,"minimal":minimal_coloring,"correct":valid_coloring and minimal_coloring}
        return json.dumps(evaluation, indent=4)
    else: raise NotImplementedError