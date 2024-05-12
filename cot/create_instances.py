import domain_utils
from fire import Fire #type: ignore

def create_instances(domain_name, num=0, overwrite_previous=False, **kwargs):
    domain = domain_utils.domains[domain_name]
    # TODO make this just fill in the stuff that's missing
    #      -> this script should handle the cartesian product thing 
    steps = range(21,30)
    tokens = range(1,5)
    total_done = 0
    overwrite_previous_flag = overwrite_previous
    for step in steps:
        for token_count in tokens:
            domain.generate_instances(num=num, overwrite_previous=overwrite_previous_flag, num_steps = step, token_length = token_count, **kwargs)
            overwrite_previous_flag = False
            total_done += num
    print(f'{total_done} instances created.')
    # TODO this should handle writing to file, and putting all the stuff inside a "raw_instance" key in each dict
    # TODO also handle randomness here, with a different pickle for each domain. Save both the sequence to generate some dataset,
    #      and the particular pickle if about to overwrite

if __name__ == "__main__":
    Fire(create_instances)