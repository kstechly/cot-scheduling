# To add a new domain, create a module in this subfolder, and then add it to the following:
from domain_utils import color_verification, coinflip, lastletterconcat, sorting, pemdas
domains = {"color_verification":color_verification, "coinflip":coinflip, "lastletterconcat":lastletterconcat, "sorting":sorting, "pemdas":pemdas}
__all__ = list(domains.keys())

# TODO refactor this to autopopulate