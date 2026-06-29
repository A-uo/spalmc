"""SpaLMC: Spot-wise Local Manifold Completion for single-cell spatial mapping."""

from .data import load_gene_list, pp_adatas
from .model import SpaLMC

pp_adata = pp_adatas

__all__ = ["SpaLMC", "pp_adatas", "pp_adata", "load_gene_list"]
