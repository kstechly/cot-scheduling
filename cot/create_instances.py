import domain_utils
from fire import Fire #type: ignore

def create_instances(domain_name, num=0, overwrite_previous=False, **kwargs):
    domain = domain_utils.domains[domain_name]
    domain.generate_instances(num=num, overwrite_previous=overwrite_previous, **kwargs)
    # TODO this should handle writing to file, and putting all the stuff inside a "raw_instance" key in each dict
    # TODO also handle randomness here, with a different pickle for each domain. Save both the sequence to generate some dataset,
    #      and the particular pickle if about to overwrite

if __name__ == "__main__":
    Fire(create_instances)